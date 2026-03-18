"""Test the MCP server path using reusable conversation fixtures.

Loads conversations from the conversations/ directory, ingests each one via
harness_ingest, then validates assertions (min nodes, search, orient) defined
in the YAML fixtures.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
# Project root is three levels up from tests/integration/external_project/
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local", override=True)

from conversation_loader import Conversation, load_all_conversations

FINDINGS: list[str] = []


def finding(msg: str) -> None:
    FINDINGS.append(msg)
    print(f"  [FINDING] {msg}")


async def test_empty_status(status_fn):
    """Verify harness_status works on an empty graph."""
    print("\n--- Testing harness_status on empty graph ---")
    try:
        result = status_fn()
        print(f"   Status output ({len(result)} chars)")
        if "0" not in result and "empty" not in result.lower():
            finding("harness_status doesn't clearly indicate empty graph")
    except Exception as e:
        finding(f"harness_status failed on empty graph: {e}")


async def test_query_views(query_fn):
    """Verify all harness_query views return non-trivial output."""
    print("\n--- Testing harness_query views ---")
    views = ["current_state", "disagreement_map", "coverage_report", "decision_log"]
    for view in views:
        try:
            result = query_fn(view=view)
            lines = result.split("\n")
            if len(lines) <= 2:
                finding(f"harness_query('{view}') returned very short output")
        except Exception as e:
            finding(f"harness_query('{view}') failed: {e}")

    # Invalid view should error clearly
    result = query_fn(view="invalid_view")
    if "Unknown" not in result and "error" not in result.lower():
        finding("harness_query with invalid view doesn't return clear error")


async def test_trace(search_fn, trace_fn):
    """Verify harness_trace works for a discovered node."""
    print("\n--- Testing harness_trace ---")
    try:
        search_result = await search_fn(query="pricing", k=1)
        id_match = re.search(r"id: ([a-f0-9]+)", search_result)
        if id_match:
            node_id = id_match.group(1)
            result = trace_fn(node_id=node_id)
            print(f"   Trace for {node_id}: {len(result)} chars")
        else:
            finding("Could not extract node ID from search results — output format may need work")
    except Exception as e:
        finding(f"harness_trace failed: {e}")

    # Invalid node ID
    result = trace_fn(node_id="nonexistent123")
    if "not found" not in result.lower():
        finding("harness_trace with invalid node_id doesn't return clear error")


async def test_debug(debug_fn):
    """Verify harness_debug returns valid JSON with expected structure."""
    print("\n--- Testing harness_debug ---")
    try:
        result = await debug_fn(
            content="User: What about freemium?\n\nAssistant: Freemium could work for the self-serve tier.",
            source="debug-test",
        )
        parsed = json.loads(result)
        print(f"   Debug: {len(parsed.get('applied', []))} applied, {len(parsed.get('dropped', []))} dropped")
    except json.JSONDecodeError:
        finding("harness_debug didn't return valid JSON")
    except Exception as e:
        finding(f"harness_debug failed: {e}")


async def ingest_conversation(conv: Conversation, ingest_fn):
    """Ingest all turns of a conversation via harness_ingest."""
    print(f"\n--- Ingesting: {conv.id} ({len(conv.turns)} turns) ---")
    for i, turn in enumerate(conv.turns, 1):
        try:
            result = await ingest_fn(content=turn.exchange, source=f"{conv.id}-turn-{i}")
            print(f"   Turn {i}: OK")
        except Exception as e:
            finding(f"[{conv.id}] Ingest turn {i} failed: {e}")


async def validate_assertions(conv: Conversation, get_engine_fn, search_fn, orient_fn):
    """Validate a conversation's assertions against the graph state."""
    print(f"\n--- Validating assertions: {conv.id} ---")
    a = conv.assertions

    engine = get_engine_fn()

    # min_nodes
    if a.min_nodes > 0:
        from cairn.models.graph_types import NodeStatus
        all_nodes = engine.graph.get_all_nodes()
        active = [n for n in all_nodes if n.status == NodeStatus.ACTIVE]
        print(f"   Nodes: {len(all_nodes)} total, {len(active)} active (need >= {a.min_nodes})")
        if len(all_nodes) < a.min_nodes:
            finding(f"[{conv.id}] Expected >= {a.min_nodes} nodes, got {len(all_nodes)}")

    # has_superseded
    if a.has_superseded:
        from cairn.models.graph_types import NodeStatus
        all_nodes = engine.graph.get_all_nodes()
        superseded = [n for n in all_nodes if n.status == NodeStatus.SUPERSEDED]
        if superseded:
            print(f"   Superseded nodes: {len(superseded)}")
            for n in superseded:
                print(f"     [{n.type.value}] {n.text[:80]}")
        else:
            finding(f"[{conv.id}] Expected superseded nodes but found none")

    # search_queries
    for sq in a.search_queries:
        try:
            result = await search_fn(query=sq.query, k=5)
            # Count non-empty result lines (rough heuristic)
            has_results = "No indexed" not in result and len(result.strip()) > 20
            if has_results:
                print(f"   Search '{sq.query}': found results")
            else:
                finding(f"[{conv.id}] Search '{sq.query}' returned no results (expected >= {sq.min_results})")
        except Exception as e:
            finding(f"[{conv.id}] Search '{sq.query}' failed: {e}")

    # orient_topics
    for ot in a.orient_topics:
        try:
            result = await orient_fn(topic=ot.topic)
            if ot.must_not_be_empty and ("No relevant" in result or len(result.strip()) < 20):
                finding(f"[{conv.id}] Orient on '{ot.topic}' returned empty/no results")
            else:
                print(f"   Orient '{ot.topic}': OK ({len(result)} chars)")
        except Exception as e:
            finding(f"[{conv.id}] Orient '{ot.topic}' failed: {e}")


async def main():
    conversations = load_all_conversations()
    print(f"Loaded {len(conversations)} test conversations")

    db_path = str(Path(__file__).parent / "test_mcp.db")

    # Clean slate
    for f in [db_path, f"{db_path}-shm", f"{db_path}-wal"]:
        if os.path.exists(f):
            os.remove(f)

    os.environ["CAIRN_DB"] = db_path

    print("=" * 60)
    print("TEST: MCP Server Path")
    print("=" * 60)

    # Import MCP server components
    print("\n1. Importing MCP server...")
    try:
        from cairn.mcp_server import _get_engine, status, query, ingest, search, orient, trace, debug
        print("   OK")
    except Exception as e:
        finding(f"MCP server import failed: {e}")
        return

    # Test empty state
    await test_empty_status(status)

    # Ingest all conversations
    for conv in conversations:
        await ingest_conversation(conv, ingest)

    # Test general MCP tool behavior
    print("\n" + "=" * 60)
    print("Testing MCP tool behavior")
    print("=" * 60)

    result = status()
    print(f"\n   Status after all ingests: {len(result)} chars")

    await test_query_views(query)
    await test_trace(search, trace)
    await test_debug(debug)

    # Validate each conversation's assertions
    print("\n" + "=" * 60)
    print("Validating conversation assertions")
    print("=" * 60)

    for conv in conversations:
        await validate_assertions(conv, _get_engine, search, orient)

    # Summary
    print("\n" + "=" * 60)
    print(f"FINDINGS: {len(FINDINGS)} issues discovered")
    print("=" * 60)
    for i, f in enumerate(FINDINGS, 1):
        print(f"  {i}. {f}")

    if not FINDINGS:
        print("  No issues found — MCP path works as documented.")

    # Cleanup
    for f in [db_path, f"{db_path}-shm", f"{db_path}-wal"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    asyncio.run(main())
