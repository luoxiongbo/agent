from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .models import AgentState


@dataclass(slots=True)
class MemoryRecord:
    id: int
    namespace: str
    kind: str
    content: str
    importance: float
    confidence: float
    metadata: dict[str, Any]
    created_at: str
    expires_at: str | None
    score: float = 0.0


class SQLiteMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memories_namespace
                ON memories(namespace);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def add(
        self,
        *,
        namespace: str,
        kind: str,
        content: str,
        importance: float = 0.7,
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> int:
        if not content.strip():
            raise ValueError("Memory content cannot be empty.")
        importance = min(max(float(importance), 0.0), 1.0)
        confidence = min(max(float(confidence), 0.0), 1.0)
        created_at = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    namespace, kind, content, importance, confidence,
                    metadata_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    kind,
                    content.strip(),
                    importance,
                    confidence,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at,
                    expires_at,
                ),
            )
            return int(cursor.lastrowid)

    def search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 6,
        minimum_confidence: float = 0.5,
    ) -> list[MemoryRecord]:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ?
                  AND confidence >= ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY importance DESC, created_at DESC
                LIMIT 200
                """,
                (namespace, minimum_confidence, now),
            ).fetchall()

        query_tokens = self._tokens(query)
        records: list[MemoryRecord] = []
        for row in rows:
            content_tokens = self._tokens(row["content"])
            overlap = len(query_tokens & content_tokens)
            union = len(query_tokens | content_tokens) or 1
            lexical = overlap / union
            score = lexical * 0.65 + float(row["importance"]) * 0.2 + float(
                row["confidence"]
            ) * 0.15
            if overlap == 0 and query_tokens:
                score *= 0.25

            records.append(
                MemoryRecord(
                    id=int(row["id"]),
                    namespace=row["namespace"],
                    kind=row["kind"],
                    content=row["content"],
                    importance=float(row["importance"]),
                    confidence=float(row["confidence"]),
                    metadata=json.loads(row["metadata_json"]),
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    score=score,
                )
            )

        records.sort(key=lambda item: item.score, reverse=True)
        return records[:limit]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]", text)
        return set(latin + chinese)

    def save_checkpoint(self, state: AgentState) -> None:
        state.touch()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(run_id, namespace, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state.run_id,
                    state.namespace,
                    json.dumps(state.to_dict(), ensure_ascii=False),
                    state.updated_at,
                ),
            )

    def load_checkpoint(self, run_id: str) -> AgentState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return AgentState.from_dict(json.loads(row["state_json"]))
