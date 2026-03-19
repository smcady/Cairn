"""Tests for VectorIndex -- sync operations and mocked async embed calls."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cairn.utils.vector_index import VectorIndex, _normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(
    embed_return: list[list[float]] | None = None,
    provider_id: str = "test-provider",
    dimensions: int = 3,
) -> MagicMock:
    """Return a MagicMock that satisfies the EmbeddingProvider protocol."""
    provider = MagicMock()
    provider.provider_id = provider_id
    provider.dimensions = dimensions
    provider.embed_documents = AsyncMock(
        return_value=embed_return or [[1.0, 0.0, 0.0]]
    )
    provider.embed_query = AsyncMock(
        return_value=(embed_return or [[1.0, 0.0, 0.0]])[0]
    )
    return provider


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

class TestCosineSimilarity:
    def _index_with(self, vectors: dict[str, list[float]]) -> VectorIndex:
        idx = VectorIndex(":memory:", provider=_mock_provider())
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
        idx = VectorIndex(":memory:", provider=_mock_provider())
        assert idx.cosine_similarity("a", "b") is None

    def test_one_missing_returns_none(self):
        idx = self._index_with({"a": [1.0, 0.0]})
        assert idx.cosine_similarity("a", "missing") is None


class TestIndexedIds:
    def test_empty(self):
        idx = VectorIndex(":memory:", provider=_mock_provider())
        assert idx.indexed_ids() == set()

    def test_contains_added(self):
        idx = VectorIndex(":memory:", provider=_mock_provider())
        idx._cache["x"] = np.array([1.0, 0.0], dtype=np.float32)
        assert "x" in idx.indexed_ids()

    def test_len(self):
        idx = VectorIndex(":memory:", provider=_mock_provider())
        idx._cache["a"] = np.array([1.0], dtype=np.float32)
        idx._cache["b"] = np.array([0.0], dtype=np.float32)
        assert len(idx) == 2


class TestRemove:
    def test_removes_from_cache_and_db(self):
        idx = VectorIndex(":memory:", provider=_mock_provider())
        vec = [0.1, 0.2, 0.3]
        idx._conn.execute(
            "INSERT INTO node_embeddings VALUES (?, ?, ?, ?)",
            ("n1", "abc", json.dumps(vec), "test-provider"),
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
        idx = VectorIndex(":memory:", provider=_mock_provider())
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
            provider = _mock_provider(provider_id="test-provider")
            idx1 = VectorIndex(db_path, provider=provider)
            idx1._conn.execute(
                "INSERT INTO node_embeddings VALUES (?, ?, ?, ?)",
                ("n1", "abc123", json.dumps(vec), "test-provider"),
            )
            idx1._conn.commit()
            idx1.close()

            idx2 = VectorIndex(db_path, provider=_mock_provider(provider_id="test-provider"))
            assert "n1" in idx2.indexed_ids()
            sim = idx2.cosine_similarity("n1", "n1")
            assert sim == pytest.approx(1.0, abs=1e-5)
        finally:
            os.unlink(db_path)

    def test_provider_switch_wipes_stale_embeddings(self):
        """Switching providers clears embeddings from the old provider."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vec = [0.1, 0.2, 0.3]
            # Create index, insert row, close -- simulates prior session
            idx1 = VectorIndex(db_path, provider=_mock_provider(provider_id="provider-A"))
            idx1._conn.execute(
                "INSERT INTO node_embeddings VALUES (?, ?, ?, ?)",
                ("n1", "abc123", json.dumps(vec), "provider-A"),
            )
            idx1._conn.commit()
            idx1.close()

            # Reopen with same provider -- should load the embedding
            idx1b = VectorIndex(db_path, provider=_mock_provider(provider_id="provider-A"))
            assert len(idx1b) == 1
            idx1b.close()

            # Reopen with different provider -- should wipe
            idx2 = VectorIndex(db_path, provider=_mock_provider(provider_id="provider-B"))
            assert len(idx2) == 0
            assert idx2.indexed_ids() == set()
            idx2.close()
        finally:
            os.unlink(db_path)

    def test_legacy_empty_provider_id_treated_as_voyage(self):
        """Rows with empty provider_id (pre-migration) are treated as Voyage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vec = [0.1, 0.2, 0.3]
            # Simulate legacy row with empty provider_id
            provider_voyage = _mock_provider(provider_id="voyage-voyage-3-lite")
            idx1 = VectorIndex(db_path, provider=provider_voyage)
            idx1._conn.execute(
                "INSERT INTO node_embeddings VALUES (?, ?, ?, ?)",
                ("n1", "abc123", json.dumps(vec), ""),  # empty = legacy
            )
            idx1._conn.commit()
            idx1.close()

            # Reopen with Voyage provider -- should NOT wipe (legacy = Voyage)
            idx2 = VectorIndex(db_path, provider=_mock_provider(provider_id="voyage-voyage-3-lite"))
            assert "n1" in idx2.indexed_ids()
            idx2.close()

            # Reopen with fastembed -- SHOULD wipe
            idx3 = VectorIndex(db_path, provider=_mock_provider(provider_id="fastembed-bge"))
            assert len(idx3) == 0
            idx3.close()
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Async operations (mocked embedding provider)
# ---------------------------------------------------------------------------

class TestAdd:
    async def test_embeds_and_caches(self):
        provider = _mock_provider(embed_return=[[1.0, 0.0, 0.0]])
        idx = VectorIndex(":memory:", provider=provider)

        await idx.add("n1", "hello world")

        assert "n1" in idx._cache
        provider.embed_documents.assert_called_once_with(["hello world"])

    async def test_skips_unchanged_text(self):
        provider = _mock_provider(embed_return=[[1.0, 0.0]])
        idx = VectorIndex(":memory:", provider=provider)

        await idx.add("n1", "hello world")
        await idx.add("n1", "hello world")  # same text, same hash

        assert provider.embed_documents.call_count == 1

    async def test_reembeds_changed_text(self):
        provider = _mock_provider(embed_return=[[1.0, 0.0]])
        idx = VectorIndex(":memory:", provider=provider)

        await idx.add("n1", "first text")
        await idx.add("n1", "different text")

        assert provider.embed_documents.call_count == 2

    async def test_skips_empty_text(self):
        provider = _mock_provider()
        idx = VectorIndex(":memory:", provider=provider)

        await idx.add("n1", "")
        await idx.add("n2", "   ")

        provider.embed_documents.assert_not_called()
        assert len(idx) == 0

    async def test_stores_provider_id_in_db(self):
        provider = _mock_provider(embed_return=[[1.0, 0.0, 0.0]], provider_id="my-provider")
        idx = VectorIndex(":memory:", provider=provider)

        await idx.add("n1", "hello")

        row = idx._conn.execute(
            "SELECT provider_id FROM node_embeddings WHERE node_id = ?", ("n1",)
        ).fetchone()
        assert row["provider_id"] == "my-provider"


class TestSearch:
    async def test_returns_sorted_by_score(self):
        provider = _mock_provider()
        provider.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])
        idx = VectorIndex(":memory:", provider=provider)

        # Pre-populate cache with known normalized vectors
        idx._cache["high"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # score ~ 1.0
        idx._cache["low"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)    # score ~ 0.0

        results = await idx.search("query", k=10)

        assert results[0][0] == "high"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
        assert results[1][0] == "low"
        assert results[1][1] == pytest.approx(0.0, abs=1e-5)

    async def test_empty_cache_returns_empty(self):
        provider = _mock_provider()
        idx = VectorIndex(":memory:", provider=provider)

        results = await idx.search("anything", k=5)
        assert results == []
        provider.embed_query.assert_not_called()

    async def test_respects_k_limit(self):
        provider = _mock_provider()
        provider.embed_query = AsyncMock(return_value=[1.0, 0.0])
        idx = VectorIndex(":memory:", provider=provider)

        for i in range(10):
            idx._cache[f"n{i}"] = np.array([1.0, 0.0], dtype=np.float32)

        results = await idx.search("q", k=3)
        assert len(results) == 3

    async def test_node_ids_filter(self):
        provider = _mock_provider()
        provider.embed_query = AsyncMock(return_value=[1.0, 0.0])
        idx = VectorIndex(":memory:", provider=provider)
        idx._cache["a"] = np.array([1.0, 0.0], dtype=np.float32)
        idx._cache["b"] = np.array([1.0, 0.0], dtype=np.float32)

        results = await idx.search("q", node_ids=["a"])
        assert all(r[0] == "a" for r in results)


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------

class TestBudget:
    async def test_budget_enforced(self):
        from cairn.utils.vector_index import EmbedBudgetError

        provider = _mock_provider(embed_return=[[1.0, 0.0]])
        idx = VectorIndex(":memory:", provider=provider, max_requests=2)

        await idx.add("n1", "text one")
        await idx.add("n2", "text two")

        with pytest.raises(EmbedBudgetError):
            await idx.add("n3", "text three")

    def test_budget_env_var_cairn(self, monkeypatch):
        """CAIRN_EMBED_MAX_REQUESTS takes precedence."""
        monkeypatch.setenv("CAIRN_EMBED_MAX_REQUESTS", "42")
        monkeypatch.setenv("VOYAGE_MAX_REQUESTS", "99")
        from cairn.utils.vector_index import _resolve_max_requests
        assert _resolve_max_requests() == 42

    def test_budget_env_var_voyage_fallback(self, monkeypatch):
        """VOYAGE_MAX_REQUESTS used as fallback."""
        monkeypatch.delenv("CAIRN_EMBED_MAX_REQUESTS", raising=False)
        monkeypatch.setenv("VOYAGE_MAX_REQUESTS", "99")
        from cairn.utils.vector_index import _resolve_max_requests
        assert _resolve_max_requests() == 99
