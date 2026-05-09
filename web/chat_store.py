"""
Persistent chat sessions for the FastAPI web UI (SQLite, WAL).

Stores alternating user/assistant messages so the server can inject dialog context into RAG
without trusting history as a source of facts (facts stay in retrieved vault chunks).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
          id TEXT PRIMARY KEY,
          title TEXT,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('user','assistant')),
          content TEXT NOT NULL,
          created_at REAL NOT NULL,
          FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);
        """
    )


class ChatStore:
    """SQLite-backed chat history (production-friendly WAL + busy timeout)."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        conn = _connect(self._path)
        try:
            init_schema(conn)
        finally:
            conn.close()

    def _cx(self) -> sqlite3.Connection:
        return _connect(self._path)

    def has_session(self, session_id: str) -> bool:
        conn = self._cx()
        try:
            cur = conn.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,))
            return cur.fetchone() is not None
        finally:
            conn.close()

    def ensure_session(self, session_id: str | None) -> str:
        """Return a valid session id; create a new session if missing or unknown."""
        if session_id:
            conn = self._cx()
            try:
                cur = conn.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,))
                if cur.fetchone():
                    return session_id
            finally:
                conn.close()
        sid = str(uuid.uuid4())
        now = time.time()
        conn = self._cx()
        try:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (sid, "Chat", now, now),
            )
        finally:
            conn.close()
        return sid

    def list_linear_messages(self, session_id: str) -> list[tuple[Role, str]]:
        """Chronological (role, content) rows for prompt / retrieval helpers."""
        conn = self._cx()
        try:
            cur = conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            out: list[tuple[Role, str]] = []
            for role, content in cur.fetchall():
                r = role if role in ("user", "assistant") else "user"
                out.append((r, content))  # type: ignore[arg-type]
            return out
        finally:
            conn.close()

    def list_messages_api(self, session_id: str, *, limit: int = 500) -> list[dict[str, str | int]]:
        """Payload for GET /api/chat/history (newest slice capped by limit)."""
        conn = self._cx()
        try:
            cur = conn.execute(
                """
                SELECT id, role, content FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max(1, min(2000, limit))),
            )
            rows = list(cur.fetchall())
        finally:
            conn.close()
        rows.reverse()
        return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

    def append_message(self, session_id: str, role: Role, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        now = time.time()
        conn = self._cx()
        try:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            if role == "user":
                title = content.replace("\n", " ").strip()
                if len(title) > 80:
                    title = title[:77] + "..."
                conn.execute(
                    "UPDATE chat_sessions SET title = ? WHERE id = ? AND title = ?",
                    (title, session_id, "Chat"),
                )
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        conn = self._cx()
        try:
            cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, str | float]]:
        lim = max(1, min(200, limit))
        conn = self._cx()
        try:
            cur = conn.execute(
                """
                SELECT id, title, created_at, updated_at FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (lim,),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
