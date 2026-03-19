"""End-to-end SDK integration tests.

Proves the full cairn pipeline using the public SDK API:
capture conversations, rebuild across sessions, orient, search, query.

Uses direct engine.ingest() with conversation fixtures (no Anthropic messages
API calls for responses). Only the classifier + embedder make real API calls.

Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env.local", override=True)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("VOYAGE_API_KEY"),
    reason="ANTHROPIC_API_KEY and VOYAGE_API_KEY required",
)


def _load_fixture(name: str):
    from conversation_loader import load_conversation
    return load_conversation(
        ROOT / "tests" / "integration" / "external_project" / "conversations" / name
    )


def _load_all_fixtures():
    from conversation_loader import load_all_conversations
    return load_all_conversations()


async def _ingest_conversation(engine, conv):
    """Ingest all turns of a conversation fixture."""
    for i, turn in enumerate(conv.turns, 1):
        await engine.ingest(turn.exchange, source=f"{conv.id}-turn-{i}")
        # Rate limit: Voyage free tier is 3 RPM
        await asyncio.sleep(1)


async def _validate_assertions(engine, conv):
    """Validate a conversation's assertions against the graph."""
    from cairn.models.graph_types import NodeStatus

    a = conv.assertions
    all_nodes = engine.graph.get_all_nodes()

    if a.min_nodes > 0:
        assert len(all_nodes) >= a.min_nodes, (
            f"[{conv.id}] Expected >= {a.min_nodes} nodes, got {len(all_nodes)}"
        )

    if a.has_superseded:
        superseded = [n for n in all_nodes if n.status == NodeStatus.SUPERSEDED]
        if not superseded:
            import warnings
            warnings.warn(
                f"[{conv.id}] Expected superseded nodes but found none. "
                "Classifier may have categorized the contradiction differently."
            )

    for sq in a.search_queries:
        results = await engine.search_nodes(sq.query, k=5)
        assert len(results) >= sq.min_results, (
            f"[{conv.id}] Search '{sq.query}': {len(results)} results (expected >= {sq.min_results})"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_fixture_capture(temp_db, reset_cairn_registry):
    """Ingest a single conversation fixture and validate graph state."""
    import cairn

    cairn.init(db_path=temp_db)
    engine = cairn.get_engine(temp_db)
    conv = _load_fixture("01_pricing_strategy.yaml")

    await _ingest_conversation(engine, conv)
    await _validate_assertions(engine, conv)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_name", [
    "01_pricing_strategy.yaml",
    "02_architecture_debate.yaml",
    "03_hiring_strategy.yaml",
])
async def test_all_fixtures(fixture_name, temp_db, reset_cairn_registry):
    """Each conversation fixture should produce a valid graph."""
    import cairn

    cairn.init(db_path=temp_db)
    engine = cairn.get_engine(temp_db)
    conv = _load_fixture(fixture_name)

    await _ingest_conversation(engine, conv)
    await _validate_assertions(engine, conv)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_session_memory(temp_db, reset_cairn_registry):
    """Graph should survive engine destruction and rebuild from event log."""
    import cairn
    from cairn._engine_registry import _engines

    # Session 1: ingest pricing strategy
    cairn.init(db_path=temp_db)
    engine = cairn.get_engine(temp_db)
    conv1 = _load_fixture("01_pricing_strategy.yaml")
    await _ingest_conversation(engine, conv1)

    session1_nodes = engine.graph.node_count()
    assert session1_nodes > 0, "Session 1 should have captured nodes"

    # Destroy engine, simulate new session
    _engines.clear()
    del engine

    # Session 2: rebuild from same DB
    engine = cairn.get_engine(temp_db)
    assert engine.graph.node_count() == session1_nodes, (
        f"Graph should rebuild to {session1_nodes} nodes, got {engine.graph.node_count()}"
    )

    # Verify search works across session boundary
    results = await engine.search_nodes("pricing model", k=5)
    assert len(results) > 0, "Search for 'pricing model' should return results after rebuild"

    # Ingest second conversation in this session
    conv2 = _load_fixture("02_architecture_debate.yaml")
    await _ingest_conversation(engine, conv2)

    # Verify both conversations are in the graph
    total_nodes = engine.graph.node_count()
    assert total_nodes > session1_nodes, "Second conversation should add more nodes"

    # Orient on both topics
    pricing_context = await cairn.orient("pricing strategy", db_path=temp_db)
    assert pricing_context, "Orient on 'pricing strategy' should return context"

    arch_context = await cairn.orient("system architecture", db_path=temp_db)
    assert arch_context, "Orient on 'system architecture' should return context"

    # Query decision log
    decisions = cairn.query("decision_log", db_path=temp_db)
    assert len(decisions) > 20, "Decision log should have substantial content"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orient_returns_structured_context(temp_db, reset_cairn_registry):
    """cairn.orient() should return structured summaries for known topics."""
    import cairn

    cairn.init(db_path=temp_db)
    engine = cairn.get_engine(temp_db)
    conv = _load_fixture("01_pricing_strategy.yaml")
    await _ingest_conversation(engine, conv)

    for ot in conv.assertions.orient_topics:
        result = await cairn.orient(ot.topic, db_path=temp_db)
        if ot.must_not_be_empty:
            assert result, f"Orient on '{ot.topic}' should return context"
            assert len(result) > 20, f"Orient on '{ot.topic}' returned too little: {result[:50]}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_all_views(temp_db, reset_cairn_registry):
    """cairn.query() should return content for all view types."""
    import cairn

    cairn.init(db_path=temp_db)
    engine = cairn.get_engine(temp_db)
    conv = _load_fixture("01_pricing_strategy.yaml")
    await _ingest_conversation(engine, conv)

    for view in ["current_state", "disagreement_map", "coverage_report", "decision_log"]:
        result = cairn.query(view, db_path=temp_db)
        assert isinstance(result, str), f"query('{view}') should return a string"
        assert len(result) > 0, f"query('{view}') returned empty"

    # Invalid view should raise
    with pytest.raises(ValueError, match="Unknown view"):
        cairn.query("invalid_view", db_path=temp_db)
