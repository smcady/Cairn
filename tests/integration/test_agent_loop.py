"""Integration test: full agent loop with automatic orient + capture.

Runs conversation fixtures through the agent loop (examples/agent_loop.py),
verifying that:
1. Conversations are captured into the graph (capture side)
2. Later turns find relevant context via orient (retrieval side)
3. Graph state matches expected assertions

Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Project root
ROOT = Path(__file__).parent.parent.parent

from dotenv import load_dotenv

load_dotenv(ROOT / ".env.local", override=True)

# Skip if API keys are missing
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("VOYAGE_API_KEY"),
    reason="ANTHROPIC_API_KEY and VOYAGE_API_KEY required",
)


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database path and clean up after."""
    db_path = str(tmp_path / "test_agent_loop.db")
    yield db_path
    for suffix in ["", "-shm", "-wal"]:
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink()


@pytest.fixture
def _reset_cairn_registry():
    """Reset the engine registry between tests."""
    from cairn._engine_registry import _engines
    _engines.clear()
    yield
    _engines.clear()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_loop_capture_and_orient(temp_db, _reset_cairn_registry):
    """Run a conversation through the agent loop and verify capture + orient."""
    import cairn
    from cairn._engine_registry import get_engine
    from cairn.integrations.anthropic import AsyncAnthropic
    from cairn.models.graph_types import NodeStatus

    # Add conversation loader to path
    sys.path.insert(0, str(ROOT / "tests" / "integration" / "external_project"))
    from conversation_loader import load_conversation

    # Add examples to path for orient_on_topic
    sys.path.insert(0, str(ROOT / "examples"))
    from agent_loop import orient_on_topic, run_turn

    cairn.init(db_path=temp_db)
    client = AsyncAnthropic()
    engine = get_engine(temp_db)
    history: list[dict] = []

    # Load the pricing strategy fixture
    fixture_path = ROOT / "tests" / "integration" / "external_project" / "conversations" / "01_pricing_strategy.yaml"
    conv = load_conversation(fixture_path)

    # Run turns through the agent loop
    orient_results = []
    for i, turn in enumerate(conv.turns, 1):
        context = await orient_on_topic(engine, turn.user)
        orient_results.append(bool(context))

        await run_turn(client, engine, turn.user, history)

        # Wait for background ingest to complete before next turn
        await asyncio.sleep(4)

    # Refresh engine state
    engine = get_engine(temp_db)
    stats = engine.get_stats()

    # Capture assertions: graph should have content
    assert stats["total_nodes"] >= conv.assertions.min_nodes, (
        f"Expected >= {conv.assertions.min_nodes} nodes, got {stats['total_nodes']}"
    )
    assert stats["total_events"] > 0, "No events captured"

    # Superseded assertions
    if conv.assertions.has_superseded:
        all_nodes = engine.graph.get_all_nodes()
        superseded = [n for n in all_nodes if n.status == NodeStatus.SUPERSEDED]
        assert len(superseded) > 0, "Expected superseded nodes but found none"

    # Search assertions
    for sq in conv.assertions.search_queries:
        results = await engine.search_nodes(sq.query, k=5)
        assert len(results) >= sq.min_results, (
            f"Search '{sq.query}': {len(results)} results (expected >= {sq.min_results})"
        )

    # Orient assertions: later turns should find context
    # Turn 1 has empty graph, but turns 2+ should find prior reasoning
    if len(conv.turns) >= 3:
        assert any(orient_results[1:]), (
            "Orient never returned context on turns after the first. "
            "The retrieval side of the loop is not working."
        )
