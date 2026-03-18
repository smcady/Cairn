"""Singleton engine management for the SDK wrapper.

Maintains one MemoryEngine per db_path so the rebuild cost is paid once,
not on every API call. Thread-safe via a module-level lock.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("cairn")

_lock = threading.Lock()
_engines: dict[str, "MemoryEngine"] = {}  # noqa: F821 — forward ref
_default_db_path: str | None = None


def init(db_path: str | os.PathLike[str] | None = None) -> None:
    """Pre-configure the default database path for Cairn.

    Call this before using the SDK wrapper if you don't want to rely on
    the CAIRN_DB environment variable.

    Args:
        db_path: Path to the SQLite database file. If None, falls back to
                 the CAIRN_DB environment variable.
    """
    global _default_db_path
    if db_path is not None:
        _default_db_path = str(Path(db_path).resolve())


def _resolve_db_path(db_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve database path from explicit arg, init() default, or env var."""
    if db_path is not None:
        return str(Path(db_path).resolve())
    if _default_db_path is not None:
        return _default_db_path
    env = os.environ.get("CAIRN_DB")
    if env:
        return str(Path(env).resolve())
    return str(Path("cairn.db").resolve())


def get_engine(db_path: str | os.PathLike[str] | None = None) -> "MemoryEngine":
    """Return a cached MemoryEngine for the given db_path, creating if needed.

    The engine is rebuilt from the event log on first access for each db_path.
    Subsequent calls with the same resolved path return the cached instance.
    """
    from cairn.memory.engine import MemoryEngine
    from cairn.models.events import EventLog
    from cairn.models.graph_types import IdeaGraph
    from cairn.utils.vector_index import VectorIndex

    resolved = _resolve_db_path(db_path)

    with _lock:
        if resolved not in _engines:
            logger.debug("cairn: initializing engine for %s", resolved)
            event_log = EventLog(resolved)
            graph = IdeaGraph()
            vector_index = VectorIndex(resolved)
            engine = MemoryEngine(
                event_log=event_log,
                graph=graph,
                vector_index=vector_index,
            )
            engine.rebuild_from_log()
            _engines[resolved] = engine

        return _engines[resolved]


def reset() -> None:
    """Clear all cached engines. Primarily for testing."""
    global _default_db_path
    with _lock:
        _engines.clear()
        _default_db_path = None
