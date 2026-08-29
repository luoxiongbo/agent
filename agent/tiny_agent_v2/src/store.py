import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class MemoryStore:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._ensure_parent_dir()
        self._init_db()

    def _ensure_parent_dir(self) -> None:
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                chroma_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        importance: float,
        chroma_id: str,
    ) -> int:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO memories
            (user_id, memory_type, content, importance, chroma_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                content,
                importance,
                chroma_id,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        memory_id = cursor.lastrowid
        conn.close()
        return memory_id

    def memory_exists(self, user_id: str, memory_type: str, content: str) -> bool:
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT 1
            FROM memories
            WHERE user_id = ? AND memory_type = ? AND content = ?
            LIMIT 1
            """,
            (user_id, memory_type, content),
        ).fetchone()
        conn.close()
        return row is not None

    def get_memories_by_chroma_ids(self, chroma_ids: List[str]) -> List[Dict]:
        if not chroma_ids:
            return []

        conn = self._get_connection()
        placeholders = ",".join(["?"] * len(chroma_ids))
        rows = conn.execute(
            f"""
            SELECT id, user_id, memory_type, content, importance, chroma_id,
                   created_at, last_accessed_at
            FROM memories
            WHERE chroma_id IN ({placeholders})
            """,
            chroma_ids,
        ).fetchall()
        conn.close()

        row_map = {row["chroma_id"]: dict(row) for row in rows}
        ordered_rows = []
        for chroma_id in chroma_ids:
            row = row_map.get(chroma_id)
            if row is not None:
                ordered_rows.append(row)
        return ordered_rows

    def list_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        conn = self._get_connection()
        if memory_type:
            rows = conn.execute(
                """
                SELECT id, user_id, memory_type, content, importance, chroma_id,
                       created_at, last_accessed_at
                FROM memories
                WHERE user_id = ? AND memory_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, memory_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, memory_type, content, importance, chroma_id,
                       created_at, last_accessed_at
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_access_time(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return

        conn = self._get_connection()
        conn.executemany(
            """
            UPDATE memories
            SET last_accessed_at = ?
            WHERE id = ?
            """,
            [
                (datetime.now().isoformat(timespec="seconds"), memory_id)
                for memory_id in memory_ids
            ],
        )
        conn.commit()
        conn.close()
