#!/usr/bin/env python3
"""
Example: agent loop with automatic cairn orientation.

Demonstrates the full capture + retrieval loop:
- Before each turn, orient on the user's message using the graph
- Inject graph context into the system prompt
- Call the API (SDK wrapper auto-captures the exchange)
- Repeat

Usage:
  Interactive:  python examples/agent_loop.py
  From fixture: python examples/agent_loop.py --fixture path/to/conversation.yaml

Requires ANTHROPIC_API_KEY in .env.local or environment.
VOYAGE_API_KEY is optional (improves embedding quality; falls back to local fastembed).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Project root is one level up from examples/
ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv

load_dotenv(ROOT / ".env.local", override=True)

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Be concise and direct."
)


async def orient_on_topic(engine, user_message: str, db_path: str | None = None) -> str:
    """Search the graph for relevant prior reasoning. Returns summary or empty string."""
    import cairn
    return await cairn.orient(user_message, k=5, db_path=db_path)


async def run_turn(
    client,
    user_message: str,
    conversation_history: list[dict],
    db_path: str | None = None,
) -> str:
    """Run one turn: orient, inject context, call API, return response."""
    context = await orient_on_topic(None, user_message, db_path=db_path)

    system = BASE_SYSTEM_PROMPT
    if context:
        system += f"\n\n## Prior reasoning context\n{context}"

    conversation_history.append({"role": "user", "content": user_message})

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=system,
        messages=conversation_history,
    )

    assistant_text = " ".join(
        b.text for b in response.content if b.type == "text"
    )
    conversation_history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


async def run_interactive(db_path: str) -> None:
    """Interactive mode: read from stdin, respond, repeat."""
    import cairn
    from cairn.integrations.anthropic import AsyncAnthropic

    cairn.init(db_path=db_path)
    client = AsyncAnthropic()
    engine = cairn.get_engine(db_path)
    history: list[dict] = []

    print("Cairn agent loop (type 'quit' to exit)")
    print(f"Graph: {engine.graph.node_count()} nodes, {engine.get_stats()['total_events']} events")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        response = await run_turn(client, user_input, history, db_path=db_path)
        print(f"Assistant: {response}\n")


async def run_fixture(fixture_path: str, db_path: str) -> dict:
    """Fixture mode: replay a conversation YAML, return stats."""
    sys.path.insert(0, str(ROOT / "tests" / "integration" / "external_project"))
    from conversation_loader import load_conversation

    import cairn
    from cairn.integrations.anthropic import AsyncAnthropic

    cairn.init(db_path=db_path)
    client = AsyncAnthropic()
    engine = cairn.get_engine(db_path)
    history: list[dict] = []

    conv = load_conversation(Path(fixture_path))
    print(f"Running fixture: {conv.id} ({len(conv.turns)} turns)")

    orient_results = []

    for i, turn in enumerate(conv.turns, 1):
        context = await orient_on_topic(None, turn.user, db_path=db_path)
        orient_results.append(bool(context))
        if context:
            print(f"  Turn {i}: [oriented] {turn.user[:60]}...")
        else:
            print(f"  Turn {i}: {turn.user[:60]}...")

        response = await run_turn(client, turn.user, history, db_path=db_path)
        print(f"  Response: {response[:80]}...")

        # Give background ingest time to complete before next turn
        await asyncio.sleep(3)

    # Final wait for any remaining background tasks
    await asyncio.sleep(3)

    # Refresh engine to see ingested events
    engine = cairn.get_engine(db_path)
    stats = engine.get_stats()
    print(f"\nGraph after: {stats['total_nodes']} nodes, {stats['total_edges']} edges, {stats['total_events']} events")
    print(f"Orient fired on turns: {[i+1 for i, r in enumerate(orient_results) if r]}")

    return {
        "stats": stats,
        "orient_results": orient_results,
        "conversation": conv,
    }


def main():
    parser = argparse.ArgumentParser(description="Cairn agent loop example")
    parser.add_argument("--fixture", help="Path to conversation YAML fixture")
    parser.add_argument("--db", default=str(ROOT / "cairn.db"), help="Database path")
    args = parser.parse_args()

    if args.fixture:
        asyncio.run(run_fixture(args.fixture, args.db))
    else:
        asyncio.run(run_interactive(args.db))


if __name__ == "__main__":
    main()
