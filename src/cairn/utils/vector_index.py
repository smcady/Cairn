"""Semantic vector index for graph nodes using Voyage AI embeddings.

Embeddings are persisted in SQLite (same file as EventLog, separate table) and cached
in memory for fast cosine similarity without redundant API calls.

Usage:
    index = VectorIndex(db_path)
    await index.add(node_id, text)          # embed + persist (no-op if text unchanged)
    results = await index.search(query, k)  # [(node_id, score), ...]
    score = index.cosine_similarity(a, b)   # sync, from in-memory cache

Budget guard:
    VectorIndex counts every Voyage AI API call (add + search). When the session total
    reaches ``max_requests``, further calls raise ``EmbedBudgetError`` immediately.
    Default cap: VOYAGE_MAX_REQUESTS env var, or 500 if unset.
    Set VOYAGE_MAX_REQUESTS=0 to disable the cap entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import voyageai

EMBEDDING_MODEL = "voyage-3-lite"

_DEFAULT_MAX_REQUESTS = 500


class EmbedBudgetError(RuntimeError):
    """Raised when the per-session Voyage AI request cap is exceeded."""


def _resolve_max_requests() -> int | None:
    """Read VOYAGE_MAX_REQUESTS from env. Returns None (unlimited) if set to 0."""
    raw = os.environ.get("VOYAGE_MAX_REQUESTS", str(_DEFAULT_MAX_REQUESTS))
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_MAX_REQUESTS
    return None if val == 0 else val


class VectorIndex:
    """SQLite-backed vector index with in-memory cosine similarity cache."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        client: voyageai.AsyncClient | None = None,
        max_requests: int | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._client = client if client is not None else voyageai.AsyncClient()
        # Billing guard: caller can override; otherwise use env var default
        self._max_requests: int | None = (
            max_requests if max_requests is not None else _resolve_max_requests()
        )
        self._request_count: int = 0
        # node_id -> unit-normalized numpy float32 vector
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    def _check_budget(self) -> None:
        """Raise EmbedBudgetError if the request cap has been reached."""
        if self._max_requests is None or self._max_requests <= 0:
            return
        if self._request_count >= self._max_requests:
            raise EmbedBudgetError(
                f"Voyage AI request cap reached ({self._request_count}/{self._max_requests}). "
                f"Set VOYAGE_MAX_REQUESTS to a higher value or 0 to disable. "
                f"Current session has made {self._request_count} embed calls."
            )

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id   TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _load_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT node_id, embedding FROM node_embeddings"
        ).fetchall()
        for row in rows:
            vec = np.array(json.loads(row["node_id" if False else "embedding"]), dtype=np.float32)
            self._cache[row["node_id"]] = _normalize(vec)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add(self, node_id: str, text: str) -> None:
        """Embed text and persist. No-op if text is unchanged since last call."""
        if not text.strip():
            return
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        row = self._conn.execute(
            "SELECT text_hash FROM node_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row and row["text_hash"] == text_hash:
            return  # already indexed with same text

        self._check_budget()
        self._request_count += 1
        result = await self._client.embed([text], model=EMBEDDING_MODEL, input_type="document")
        embedding: list[float] = result.embeddings[0]

        self._conn.execute(
            """INSERT OR REPLACE INTO node_embeddings (node_id, text_hash, embedding)
               VALUES (?, ?, ?)""",
            (node_id, text_hash, json.dumps(embedding)),
        )
        self._conn.commit()
        self._cache[node_id] = _normalize(np.array(embedding, dtype=np.float32))

    def remove(self, node_id: str) -> None:
        """Remove a node's embedding (e.g. after supersede)."""
        self._conn.execute("DELETE FROM node_embeddings WHERE node_id = ?", (node_id,))
        self._conn.commit()
        self._cache.pop(node_id, None)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        k: int = 10,
        node_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k (node_id, cosine_score) for a query string.

        Args:
            query: text to search for
            k: number of results to return
            node_ids: restrict search to this subset; None searches all indexed nodes
        """
        if not self._cache:
            return []

        self._check_budget()
        self._request_count += 1
        result = await self._client.embed([query], model=EMBEDDING_MODEL, input_type="query")
        query_vec = _normalize(np.array(result.embeddings[0], dtype=np.float32))

        candidates = node_ids if node_ids is not None else list(self._cache.keys())
        scores: list[tuple[str, float]] = []
        for nid in candidates:
            vec = self._cache.get(nid)
            if vec is None:
                continue
            scores.append((nid, float(np.dot(query_vec, vec))))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def cosine_similarity(self, id_a: str, id_b: str) -> float | None:
        """Synchronous pairwise similarity from cache. Returns None if either not indexed."""
        vec_a = self._cache.get(id_a)
        vec_b = self._cache.get(id_b)
        if vec_a is None or vec_b is None:
            return None
        return float(np.dot(vec_a, vec_b))

    def indexed_ids(self) -> set[str]:
        return set(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._conn.close()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec
