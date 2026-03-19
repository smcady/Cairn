"""Semantic vector index for graph nodes.

Embeddings are persisted in SQLite (same file as EventLog, separate table) and cached
in memory for fast cosine similarity without redundant API calls.

Supports pluggable embedding providers via the EmbeddingProvider protocol.
Default provider is auto-detected: Voyage AI if VOYAGE_API_KEY is set,
otherwise local fastembed (no API key needed).

Usage:
    index = VectorIndex(db_path)
    await index.add(node_id, text)          # embed + persist (no-op if text unchanged)
    results = await index.search(query, k)  # [(node_id, score), ...]
    score = index.cosine_similarity(a, b)   # sync, from in-memory cache

Budget guard:
    VectorIndex counts every embedding API call (add + search). When the session total
    reaches ``max_requests``, further calls raise ``EmbedBudgetError`` immediately.
    Default cap: CAIRN_EMBED_MAX_REQUESTS or VOYAGE_MAX_REQUESTS env var, or 500 if unset.
    Set the env var to 0 to disable the cap entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path

import numpy as np

from cairn.utils.embedding_providers import EmbeddingProvider, get_default_provider

logger = logging.getLogger("cairn")

_DEFAULT_MAX_REQUESTS = 500

# Legacy provider_id for rows created before provider tracking was added.
# These were always created by voyage-3-lite.
_LEGACY_PROVIDER_ID = "voyage-voyage-3-lite"


class EmbedBudgetError(RuntimeError):
    """Raised when the per-session embedding request cap is exceeded."""


def _resolve_max_requests() -> int | None:
    """Read embed budget from env. Returns None (unlimited) if set to 0."""
    raw = os.environ.get(
        "CAIRN_EMBED_MAX_REQUESTS",
        os.environ.get("VOYAGE_MAX_REQUESTS", str(_DEFAULT_MAX_REQUESTS)),
    )
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
        provider: EmbeddingProvider | None = None,
        max_requests: int | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._provider = provider if provider is not None else get_default_provider()
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
                f"Embedding request cap reached ({self._request_count}/{self._max_requests}). "
                f"Set CAIRN_EMBED_MAX_REQUESTS to a higher value or 0 to disable. "
                f"Current session has made {self._request_count} embed calls."
            )

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id   TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedding TEXT NOT NULL,
                provider_id TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.commit()
        # Migration: add provider_id column to existing tables
        try:
            self._conn.execute(
                "ALTER TABLE node_embeddings ADD COLUMN provider_id TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def _load_cache(self) -> None:
        """Load embeddings from SQLite into memory. Wipes stale embeddings from a different provider."""
        current_id = self._provider.provider_id

        # Check for provider mismatch
        row = self._conn.execute(
            "SELECT provider_id FROM node_embeddings LIMIT 1"
        ).fetchone()
        if row is not None:
            stored_id = row["provider_id"]
            # Treat empty string as legacy Voyage embeddings
            effective_stored = stored_id if stored_id else _LEGACY_PROVIDER_ID
            if effective_stored != current_id:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM node_embeddings"
                ).fetchone()[0]
                logger.warning(
                    "cairn: embedding provider changed (%s -> %s). "
                    "Clearing %d stale embeddings (will re-embed on next access).",
                    effective_stored, current_id, count,
                )
                self._conn.execute("DELETE FROM node_embeddings")
                self._conn.commit()
                return

        rows = self._conn.execute(
            "SELECT node_id, embedding FROM node_embeddings"
        ).fetchall()
        for row in rows:
            vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
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
        vectors = await self._provider.embed_documents([text])
        embedding: list[float] = vectors[0]

        self._conn.execute(
            """INSERT OR REPLACE INTO node_embeddings (node_id, text_hash, embedding, provider_id)
               VALUES (?, ?, ?, ?)""",
            (node_id, text_hash, json.dumps(embedding), self._provider.provider_id),
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
        query_embedding = await self._provider.embed_query(query)
        query_vec = _normalize(np.array(query_embedding, dtype=np.float32))

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
