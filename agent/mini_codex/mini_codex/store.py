"""SQLite checkpoints and event history, with a workspace-wide process lock."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, workspace: Path):
        directory = workspace / ".mini-codex"
        if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
            raise ValueError("Session directory must not be a symlink or junction")
        directory.mkdir(exist_ok=True)
        self.directory = directory
        db_path = directory / "sessions.sqlite3"
        if db_path.is_symlink():
            raise ValueError("Session database must not be a symlink")
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, status TEXT NOT NULL,
                messages TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                created_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)

    @contextmanager
    def exclusive(self):
        path = self.directory / "runtime.lock"
        if path.is_symlink():
            raise ValueError("Lock file must not be a symlink")
        with path.open("a+b") as stream:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("Another Mini Codex process is using this workspace") from None
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def create(self) -> str:
        sid = uuid.uuid4().hex
        self.save(sid, [], "idle", "session_created", {})
        return sid

    def save(self, sid: str, messages: list[dict], status: str, kind: str, payload: dict):
        # Event + checkpoint commit together, so readers cannot see half a transition.
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status, messages=excluded.messages",
                            (sid, now(), status, json.dumps(messages, ensure_ascii=False)))
            self.db.execute("INSERT INTO events(session_id, created_at, kind, payload) VALUES (?, ?, ?, ?)",
                            (sid, now(), kind, json.dumps(payload, ensure_ascii=False)))

    def load(self, sid: str) -> list[dict]:
        row = self.db.execute("SELECT messages FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise ValueError("Session not found: " + sid)
        return json.loads(row[0])

    def list_sessions(self) -> list[dict]:
        rows = self.db.execute("SELECT id, updated_at, status FROM sessions ORDER BY updated_at DESC LIMIT 50").fetchall()
        return [{"id": row[0], "updated_at": row[1], "status": row[2]} for row in rows]

    def close(self):
        self.db.close()
