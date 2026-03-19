"""Test the SDK wrapper path using reusable conversation fixtures.

Loads conversations from the conversations/ directory, runs each through the
drop-in AsyncAnthropic wrapper, then validates assertions against the graph.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from dotenv import load_dotenv
# Project root is three levels up from tests/integration/external_project/
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local", override=True)

from conversation_loader import Conversation, load_all_conversations

FINDINGS: list[str] = []


def finding(msg: str) -> None:
    FINDINGS.append(msg)
    print(f"  [FINDING] {msg}")


async def run_conversation(conv: Conversation, client):
    """Run a conversation through the SDK wrapper (real API calls)."""
    print(f"\n--- Running: {conv.id} ({len(conv.turns)} turns) ---")
    for i, turn in enumerate(conv.turns, 1):
        print(f"   Turn {i}: {turn.user[:60]}...")
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                messages=[{"role": "user", "content": turn.user}],
            )
            assistant_text = " ".join(
                b.text for b in response.content if b.type == "text"
            )
            print(f"   Response: {assistant_text[:80]}...")
        except Exception as e:
            finding(f"[{conv.id}] Turn {i} API call failed: {e}")
            return False
    return True


async def validate_assertions(conv: Conversation, engine):
    """Validate a conversation's assertions against the graph."""
    print(f"\n--- Validating: {conv.id} ---")
    a = conv.assertions

    from cairn.models.graph_types import NodeStatus
    all_nodes = engine.graph.get_all_nodes()
    active = [n for n in all_nodes if n.status == NodeStatus.ACTIVE]
    superseded = [n for n in all_nodes if n.status == NodeStatus.SUPERSEDED]

    # min_nodes (cumulative across all conversations ingested so far)
    if a.min_nodes > 0:
        print(f"   Nodes: {len(all_nodes)} total, {len(active)} active")

    # has_superseded
    if a.has_superseded:
        if superseded:
            print(f"   Superseded: {len(superseded)}")
            for n in superseded:
                print(f"     [{n.type.value}] {n.text[:80]}")
        else:
            finding(f"[{conv.id}] Expected superseded nodes but found none")

    # search_queries
    for sq in a.search_queries:
        try:
            results = await engine.search_nodes(sq.query, k=5)
            if len(results) >= sq.min_results:
                print(f"   Search '{sq.query}': {len(results)} results")
            else:
                finding(f"[{conv.id}] Search '{sq.query}': {len(results)} results (expected >= {sq.min_results})")
        except Exception as e:
            finding(f"[{conv.id}] Search '{sq.query}' failed: {e}")

    # orient_topics — SDK path doesn't have orient directly, so we use search as proxy
    for ot in a.orient_topics:
        try:
            results = await engine.search_nodes(ot.topic, k=5)
            if ot.must_not_be_empty and not results:
                finding(f"[{conv.id}] No results for orient topic '{ot.topic}'")
            else:
                print(f"   Orient topic '{ot.topic}': {len(results)} related nodes")
        except Exception as e:
            finding(f"[{conv.id}] Orient topic '{ot.topic}' failed: {e}")


async def main():
    conversations = load_all_conversations()
    print(f"Loaded {len(conversations)} test conversations")

    db_path = str(Path(__file__).parent / "test_sdk.db")

    # Clean slate
    for f in [db_path, f"{db_path}-shm", f"{db_path}-wal"]:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 60)
    print("TEST: SDK Wrapper Path")
    print("=" * 60)

    # Import and configure
    print("\n1. Import and configure cairn...")
    try:
        import cairn
        cairn.init(db_path=db_path)
        print("   cairn.init() OK")
    except Exception as e:
        finding(f"cairn.init() failed: {e}")
        return

    try:
        from cairn.integrations.anthropic import AsyncAnthropic
        print("   Import AsyncAnthropic OK")
    except Exception as e:
        finding(f"Import failed: {e}")
        return

    # Create client
    print("\n2. Create AsyncAnthropic client...")
    try:
        client = AsyncAnthropic()
        print("   Client created OK")
    except Exception as e:
        finding(f"Client creation failed: {e}")
        return

    # Run all conversations
    for conv in conversations:
        success = await run_conversation(conv, client)
        if not success:
            print(f"   Skipping remaining turns for {conv.id}")

    # Wait for background ingest tasks
    print("\n   Waiting for background ingest tasks...")
    await asyncio.sleep(5)

    # Get engine for assertions
    print("\n3. Checking graph state...")
    try:
        from cairn._engine_registry import get_engine
        engine = get_engine(db_path)
        stats = engine.get_stats()
        print(f"   Graph stats: {stats}")
    except Exception as e:
        finding(f"Graph check failed: {e}")
        return

    # Validate assertions for each conversation
    print("\n" + "=" * 60)
    print("Validating conversation assertions")
    print("=" * 60)

    for conv in conversations:
        await validate_assertions(conv, engine)

    # Summary
    print("\n" + "=" * 60)
    print(f"FINDINGS: {len(FINDINGS)} issues discovered")
    print("=" * 60)
    for i, f in enumerate(FINDINGS, 1):
        print(f"  {i}. {f}")

    if not FINDINGS:
        print("  No issues found — SDK path works as documented.")

    # Cleanup
    for f in [db_path, f"{db_path}-shm", f"{db_path}-wal"]:
        if os.path.exists(f):
            os.remove(f)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sdk_full_pipeline():
    """Pytest-collected wrapper: run full SDK pipeline and assert no findings."""
    await main()
    assert not FINDINGS, f"SDK pipeline had {len(FINDINGS)} findings:\n" + "\n".join(FINDINGS)


if __name__ == "__main__":
    asyncio.run(main())
