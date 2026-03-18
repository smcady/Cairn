"""WorkspaceRegistry — manages named idea workspaces backed by SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Workspace:
    id: str
    label: str
    created_at: str


class WorkspaceRegistry:
    """CRUD registry for workspaces, backed by a shared SQLite connection."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is not None:
            self._conn = conn
        else:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()
        # Ensure default workspace always exists (for legacy nodes with workspace_id="")
        self._ensure_default()

    def _ensure_default(self) -> None:
        existing = self._conn.execute(
            "SELECT id FROM workspaces WHERE id = 'default'"
        ).fetchone()
        if existing is None:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO workspaces (id, label, created_at) VALUES ('default', 'Default', ?)",
                (now,),
            )
            self._conn.commit()

    def create(self, label: str = "") -> Workspace:
        ws_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO workspaces (id, label, created_at) VALUES (?, ?, ?)",
            (ws_id, label, now),
        )
        self._conn.commit()
        return Workspace(id=ws_id, label=label, created_at=now)

    def get(self, ws_id: str) -> Workspace | None:
        row = self._conn.execute(
            "SELECT id, label, created_at FROM workspaces WHERE id = ?", (ws_id,)
        ).fetchone()
        if row is None:
            return None
        return Workspace(id=row["id"], label=row["label"], created_at=row["created_at"])

    def get_all(self) -> list[Workspace]:
        rows = self._conn.execute(
            "SELECT id, label, created_at FROM workspaces ORDER BY created_at ASC"
        ).fetchall()
        return [Workspace(id=r["id"], label=r["label"], created_at=r["created_at"]) for r in rows]

    def update_label(self, ws_id: str, label: str) -> None:
        self._conn.execute(
            "UPDATE workspaces SET label = ? WHERE id = ?", (label, ws_id)
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()
        return row[0]
