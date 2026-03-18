"""Tests for VectorIndex — sync operations and mocked async embed calls."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from cairn.utils.vector_index import VectorIndex, _normalize


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _normalize(v)
        np.testing.assert_allclose(result, v)

    def test_normalizes_to_unit_length(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector_safe(self):
        v = np.array([0.0, 0.0], dtype=np.float32)
        result = _normalize(v)
        # Should not raise, returns original
        assert result is not None


# ---------------------------------------------------------------------------
# Sync cache operations (no API calls)
# ---------------------------------------------------------------------------

def _mock_client() -> MagicMock:
    """Return a MagicMock that stands in for voyageai.AsyncClient without needing credentials."""
    return MagicMock()


class TestCosineSimilarity:
    def _index_with(self, vectors: dict[str, list[float]]) -> VectorIndex:
        idx = VectorIndex(":memory:", client=_mock_client())
        for node_id, vec in vectors.items():
            arr = np.array(vec, dtype=np.float32)
            idx._cache[node_id] = _normalize(arr)
        return idx

    def test_identical_vectors(self):
        idx = self._index_with({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        assert idx.cosine_similarity("a", "b") == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        idx = self._index_with({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert idx.cosine_similarity("a", "b") == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        idx = self._index_with({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
        assert idx.cosine_similarity("a", "b") == pytest.approx(-1.0, abs=1e-5)

    def test_missing_node_returns_none(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        assert idx.cosine_similarity("a", "b") is None

    def test_one_missing_returns_none(self):
        idx = self._index_with({"a": [1.0, 0.0]})
        assert idx.cosine_similarity("a", "missing") is None


class TestIndexedIds:
    def test_empty(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        assert idx.indexed_ids() == set()

    def test_contains_added(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._cache["x"] = np.array([1.0, 0.0], dtype=np.float32)
        assert "x" in idx.indexed_ids()

    def test_len(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._cache["a"] = np.array([1.0], dtype=np.float32)
        idx._cache["b"] = np.array([0.0], dtype=np.float32)
        assert len(idx) == 2


class TestRemove:
    def test_removes_from_cache_and_db(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        vec = [0.1, 0.2, 0.3]
        idx._conn.execute(
            "INSERT INTO node_embeddings VALUES (?, ?, ?)",
            ("n1", "abc", json.dumps(vec)),
        )
        idx._conn.commit()
        idx._cache["n1"] = np.array(vec, dtype=np.float32)

        idx.remove("n1")

        assert "n1" not in idx._cache
        row = idx._conn.execute(
            "SELECT * FROM node_embeddings WHERE node_id = ?", ("n1",)
        ).fetchone()
        assert row is None

    def test_remove_nonexistent_is_safe(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx.remove("does_not_exist")  # should not raise


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_embeddings_survive_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vec = [0.1, 0.2, 0.3]
            idx1 = VectorIndex(db_path, client=_mock_client())
            idx1._conn.execute(
                "INSERT INTO node_embeddings VALUES (?, ?, ?)",
                ("n1", "abc123", json.dumps(vec)),
            )
            idx1._conn.commit()
            idx1.close()

            idx2 = VectorIndex(db_path, client=_mock_client())
            assert "n1" in idx2.indexed_ids()
            sim = idx2.cosine_similarity("n1", "n1")
            assert sim == pytest.approx(1.0, abs=1e-5)
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Async operations (mocked Voyage AI client)
# ---------------------------------------------------------------------------

def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    mock = MagicMock()
    mock.embeddings = vectors
    return mock


class TestAdd:
    async def test_embeds_and_caches(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0, 0.0]])

        await idx.add("n1", "hello world")

        assert "n1" in idx._cache
        idx._client.embed.assert_called_once()

    async def test_skips_unchanged_text(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0]])

        await idx.add("n1", "hello world")
        await idx.add("n1", "hello world")  # same text, same hash

        assert idx._client.embed.call_count == 1

    async def test_reembeds_changed_text(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0]])

        await idx.add("n1", "first text")
        await idx.add("n1", "different text")

        assert idx._client.embed.call_count == 2

    async def test_skips_empty_text(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()

        await idx.add("n1", "")
        await idx.add("n2", "   ")

        idx._client.embed.assert_not_called()
        assert len(idx) == 0


class TestSearch:
    async def test_returns_sorted_by_score(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        # Query vector: [1, 0, 0]
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0, 0.0]])

        # Pre-populate cache with known normalized vectors
        idx._cache["high"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # score ≈ 1.0
        idx._cache["low"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)    # score ≈ 0.0

        results = await idx.search("query", k=10)

        assert results[0][0] == "high"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
        assert results[1][0] == "low"
        assert results[1][1] == pytest.approx(0.0, abs=1e-5)

    async def test_empty_cache_returns_empty(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()

        results = await idx.search("anything", k=5)
        assert results == []
        idx._client.embed.assert_not_called()

    async def test_respects_k_limit(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0]])

        for i in range(10):
            idx._cache[f"n{i}"] = np.array([1.0, 0.0], dtype=np.float32)

        results = await idx.search("q", k=3)
        assert len(results) == 3

    async def test_node_ids_filter(self):
        idx = VectorIndex(":memory:", client=_mock_client())
        idx._client = AsyncMock()
        idx._client.embed.return_value = _make_embed_response([[1.0, 0.0]])
        idx._cache["a"] = np.array([1.0, 0.0], dtype=np.float32)
        idx._cache["b"] = np.array([1.0, 0.0], dtype=np.float32)

        results = await idx.search("q", node_ids=["a"])
        assert all(r[0] == "a" for r in results)
