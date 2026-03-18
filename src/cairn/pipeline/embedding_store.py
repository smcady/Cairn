"""EmbeddingStore — Voyage AI embeddings with SQLite BLOB storage."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import voyageai
    _VOYAGE_AVAILABLE = True
except ImportError:
    _VOYAGE_AVAILABLE = False

if TYPE_CHECKING:
    from cairn.models.graph_types import IdeaGraph

_VOYAGE_MODEL = "voyage-3-lite"
_ROUTING_THRESHOLD = 0.65
_ROUTING_TOP_K = 5  # top-K nodes per workspace for routing avg


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity between two 1-D vectors."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _serialize(arr: "np.ndarray") -> bytes:
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _deserialize(blob: bytes) -> "np.ndarray":
    import numpy as np
    return np.load(io.BytesIO(blob))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingStore:
    """Async embedding store backed by Voyage AI and SQLite BLOB storage."""

    def __init__(
        self,
        api_key: str,
        db_path: str | Path = ":memory:",
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not _NUMPY_AVAILABLE:
            raise ImportError("numpy is required for EmbeddingStore")
        if not _VOYAGE_AVAILABLE:
            raise ImportError("voyageai is required for EmbeddingStore")

        self._api_key = api_key
        self._voyage_client: voyageai.AsyncClient | None = None

        if conn is not None:
            self._conn = conn
        else:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row

        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                text_hash TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _get_client(self) -> "voyageai.AsyncClient":
        if self._voyage_client is None:
            self._voyage_client = voyageai.AsyncClient(api_key=self._api_key)
        return self._voyage_client

    async def _embed(self, texts: list[str]) -> "list[np.ndarray]":
        """Call Voyage AI to embed a list of texts."""
        import numpy as np
        client = self._get_client()
        result = await client.embed(texts, model=_VOYAGE_MODEL, input_type="document")
        return [np.array(v, dtype=np.float32) for v in result.embeddings]

    async def upsert(self, node_id: str, text: str, workspace_id: str) -> None:
        """Embed and store a node. Skips if text hash is unchanged."""
        new_hash = _text_hash(text)
        row = self._conn.execute(
            "SELECT text_hash FROM node_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is not None and row["text_hash"] == new_hash:
            return  # unchanged

        embeddings = await self._embed([text])
        blob = _serialize(embeddings[0])
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO node_embeddings (node_id, embedding, text_hash, workspace_id, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   embedding=excluded.embedding,
                   text_hash=excluded.text_hash,
                   workspace_id=excluded.workspace_id,
                   updated_at=excluded.updated_at""",
            (node_id, blob, new_hash, workspace_id, now),
        )
        self._conn.commit()

    async def upsert_batch(self, items: list[tuple[str, str, str]]) -> None:
        """Batch upsert: items = [(node_id, text, workspace_id), ...].

        Deduplicates by checking text_hash and makes one API call for all new/changed nodes.
        """
        to_embed: list[tuple[str, str, str]] = []  # (node_id, text, workspace_id)

        for node_id, text, workspace_id in items:
            new_hash = _text_hash(text)
            row = self._conn.execute(
                "SELECT text_hash FROM node_embeddings WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None or row["text_hash"] != new_hash:
                to_embed.append((node_id, text, workspace_id))

        if not to_embed:
            return

        texts = [item[1] for item in to_embed]
        embeddings = await self._embed(texts)

        now = datetime.now(timezone.utc).isoformat()
        for (node_id, text, workspace_id), embedding in zip(to_embed, embeddings):
            blob = _serialize(embedding)
            new_hash = _text_hash(text)
            self._conn.execute(
                """INSERT INTO node_embeddings (node_id, embedding, text_hash, workspace_id, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET
                       embedding=excluded.embedding,
                       text_hash=excluded.text_hash,
                       workspace_id=excluded.workspace_id,
                       updated_at=excluded.updated_at""",
                (node_id, blob, new_hash, workspace_id, now),
            )
        self._conn.commit()

    def get(self, node_id: str) -> "np.ndarray | None":
        """Return the stored embedding for a node, or None if not present."""
        row = self._conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return _deserialize(row["embedding"])

    async def find_nearest(
        self,
        query_text: str,
        top_k: int = 8,
        workspace_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find the top_k nearest nodes to query_text.

        Returns a list of (node_id, similarity) sorted descending by similarity.
        When workspace_id is provided, only nodes from that workspace are considered.
        """
        import numpy as np

        embeddings_q = await self._embed([query_text])
        query_vec = embeddings_q[0]

        if workspace_id is not None:
            rows = self._conn.execute(
                "SELECT node_id, embedding FROM node_embeddings WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT node_id, embedding FROM node_embeddings"
            ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            vec = _deserialize(row["embedding"])
            sim = _cosine_similarity(query_vec, vec)
            results.append((row["node_id"], sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def route_input(
        self,
        text: str,
        graph: "IdeaGraph",
    ) -> tuple[str | None, float]:
        """Route input text to the best-matching workspace.

        For each workspace (excluding legacy '' workspace_id rows), compute the
        average cosine similarity of the top-5 nearest nodes. Return the workspace
        with the highest average if it exceeds the threshold (0.65).

        Returns (workspace_id, confidence) or (None, 0.0) if below threshold
        or no workspaces have embeddings.
        """
        import numpy as np

        # Get all distinct workspace_ids with embeddings (skip legacy empty workspace_id)
        rows = self._conn.execute(
            "SELECT DISTINCT workspace_id FROM node_embeddings WHERE workspace_id != ''"
        ).fetchall()
        workspace_ids = [r["workspace_id"] for r in rows]

        if not workspace_ids:
            return None, 0.0

        embeddings_q = await self._embed([text])
        query_vec = embeddings_q[0]

        best_ws: str | None = None
        best_score = 0.0

        for ws_id in workspace_ids:
            ws_rows = self._conn.execute(
                "SELECT node_id, embedding FROM node_embeddings WHERE workspace_id = ?",
                (ws_id,),
            ).fetchall()
            if not ws_rows:
                continue

            sims = []
            for row in ws_rows:
                vec = _deserialize(row["embedding"])
                sims.append(_cosine_similarity(query_vec, vec))

            sims.sort(reverse=True)
            top_sims = sims[:_ROUTING_TOP_K]
            avg_sim = float(np.mean(top_sims))

            if avg_sim > best_score:
                best_score = avg_sim
                best_ws = ws_id

        if best_score >= _ROUTING_THRESHOLD:
            return best_ws, best_score

        return None, best_score

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()
        return row[0]

    def count_without_embedding(self, graph: "IdeaGraph") -> int:
        """Count how many graph nodes have no stored embedding."""
        all_ids = {n.id for n in graph.get_all_nodes()}
        stored_ids = {
            r["node_id"]
            for r in self._conn.execute("SELECT node_id FROM node_embeddings").fetchall()
        }
        return len(all_ids - stored_ids)
