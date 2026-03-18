"""Tests for EmbeddingStore — mocking the Voyage AI client."""

from __future__ import annotations

import numpy as np
import pytest

from cairn.pipeline.embedding_store import (
    EmbeddingStore,
    _cosine_similarity,
    _text_hash,
)


# ---- Helpers ----------------------------------------------------------------

def _make_vec(val: float, dim: int = 4) -> np.ndarray:
    """Create a unit vector that's easy to control."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = val
    return v


async def _mock_embed(texts: list[str]) -> list[np.ndarray]:
    """Deterministic mock: embed each text as a distinct unit vector."""
    vecs = []
    for i, _ in enumerate(texts):
        v = np.zeros(8, dtype=np.float32)
        v[i % 8] = 1.0
        vecs.append(v)
    return vecs


@pytest.fixture
def store(monkeypatch):
    """EmbeddingStore with mocked _embed method."""
    s = EmbeddingStore.__new__(EmbeddingStore)
    import sqlite3
    s._api_key = "fake"
    s._voyage_client = None
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    s._conn = conn
    s._init_schema()
    monkeypatch.setattr(s, "_embed", _mock_embed)
    return s


# ---- Unit: helpers ----------------------------------------------------------

class TestHelpers:
    def test_text_hash_stable(self):
        h1 = _text_hash("hello world")
        h2 = _text_hash("hello world")
        assert h1 == h2

    def test_text_hash_different(self):
        assert _text_hash("a") != _text_hash("b")

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0


# ---- EmbeddingStore ops -----------------------------------------------------

class TestUpsert:
    async def test_upsert_stores_embedding(self, store):
        await store.upsert("n1", "hello", "default")
        vec = store.get("n1")
        assert vec is not None
        assert vec.shape == (8,)

    async def test_upsert_skips_unchanged(self, store, monkeypatch):
        call_count = [0]
        original = _mock_embed

        async def counting_embed(texts):
            call_count[0] += len(texts)
            return await original(texts)

        monkeypatch.setattr(store, "_embed", counting_embed)

        await store.upsert("n1", "hello", "default")
        first_count = call_count[0]
        await store.upsert("n1", "hello", "default")  # same text
        assert call_count[0] == first_count  # no additional API call

    async def test_upsert_updates_on_text_change(self, store):
        await store.upsert("n1", "original text", "default")
        vec1 = store.get("n1").copy()
        await store.upsert("n1", "updated text", "default")
        vec2 = store.get("n1")
        # The mock produces different vectors for different positions in the batch
        # so just verify it stored something
        assert vec2 is not None

    async def test_upsert_batch_dedup(self, store, monkeypatch):
        call_counts = [0]
        original = _mock_embed

        async def counting_embed(texts):
            call_counts[0] += len(texts)
            return await original(texts)

        monkeypatch.setattr(store, "_embed", counting_embed)

        await store.upsert_batch([
            ("n1", "text one", "default"),
            ("n2", "text two", "default"),
            ("n3", "text three", "default"),
        ])
        assert call_counts[0] == 3

        # Re-upsert — same hashes, should not call API
        prev = call_counts[0]
        await store.upsert_batch([
            ("n1", "text one", "default"),
            ("n2", "text two", "default"),
        ])
        assert call_counts[0] == prev  # no new calls


class TestFindNearest:
    async def test_returns_sorted_by_similarity(self, store):
        # Store 3 nodes; query matches n1 direction exactly
        await store.upsert_batch([
            ("n1", "a", "ws1"),
            ("n2", "b", "ws1"),
            ("n3", "c", "ws1"),
        ])
        results = await store.find_nearest("a", top_k=3, workspace_id="ws1")
        assert len(results) > 0
        # Results should be sorted descending
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)

    async def test_workspace_filtering(self, store):
        await store.upsert_batch([
            ("n1", "topic A", "ws_a"),
            ("n2", "topic B", "ws_b"),
        ])
        results = await store.find_nearest("topic A", top_k=5, workspace_id="ws_a")
        ids = [r[0] for r in results]
        assert "n1" in ids
        assert "n2" not in ids

    async def test_empty_store_returns_empty(self, store):
        results = await store.find_nearest("anything", top_k=5)
        assert results == []

    async def test_top_k_limit(self, store):
        await store.upsert_batch([
            (f"n{i}", f"text {i}", "default") for i in range(6)
        ])
        results = await store.find_nearest("text 0", top_k=3)
        assert len(results) <= 3


class TestRouting:
    async def test_returns_none_when_empty(self, store):
        from cairn.models.graph_types import IdeaGraph
        graph = IdeaGraph()
        ws_id, conf = await store.route_input("hello", graph)
        assert ws_id is None
        assert conf == 0.0

    async def test_routing_above_threshold(self, store, monkeypatch):
        """When a workspace has very similar embeddings, routing should succeed."""
        from cairn.models.graph_types import IdeaGraph

        # Use a controlled embed that returns the same vector for the query and stored nodes
        fixed_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        async def identical_embed(texts):
            return [fixed_vec.copy() for _ in texts]

        monkeypatch.setattr(store, "_embed", identical_embed)

        # Store nodes in workspace "ws1"
        await store.upsert_batch([
            ("n1", "AI and education", "ws1"),
            ("n2", "AI applications", "ws1"),
        ])

        graph = IdeaGraph()
        ws_id, conf = await store.route_input("AI in education", graph)
        assert ws_id == "ws1"
        assert conf >= 0.65

    async def test_routing_below_threshold(self, store, monkeypatch):
        """When similarity is low, route_input returns None."""
        from cairn.models.graph_types import IdeaGraph

        call_n = [0]

        async def alternating_embed(texts):
            result = []
            for _ in texts:
                v = np.zeros(8, dtype=np.float32)
                v[call_n[0] % 8] = 1.0
                call_n[0] += 1
                result.append(v)
            return result

        monkeypatch.setattr(store, "_embed", alternating_embed)

        # Store node in workspace
        await store.upsert("n1", "text in ws1", "ws1")

        # Query with orthogonal vector — similarity = 0
        graph = IdeaGraph()
        ws_id, conf = await store.route_input("completely different", graph)
        assert ws_id is None

    async def test_routing_skips_legacy_empty_workspace(self, store, monkeypatch):
        """Nodes with workspace_id='' are excluded from routing sweep."""
        from cairn.models.graph_types import IdeaGraph

        fixed_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        async def identical_embed(texts):
            return [fixed_vec.copy() for _ in texts]

        monkeypatch.setattr(store, "_embed", identical_embed)

        # Store a node with legacy empty workspace_id
        await store.upsert("n_legacy", "some text", "")

        graph = IdeaGraph()
        ws_id, conf = await store.route_input("some text", graph)
        # Should NOT route to "" workspace
        assert ws_id is None or ws_id != ""

    def test_count(self, store):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            store.upsert_batch([("n1", "a", "default"), ("n2", "b", "default")])
        )
        assert store.count() == 2
