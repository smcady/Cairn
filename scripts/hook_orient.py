#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook: inject graph context before the model responds.

Reads stdin JSON from Claude Code (contains the user's prompt), searches the
cairn graph for relevant prior reasoning, and prints the result to stdout as
additionalContext. The model sees this context before it starts thinking.

If the graph is empty or no relevant results are found, exits silently.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv

load_dotenv(ROOT / ".env.local", override=True)

# Ensure CAIRN_DB has a sensible default
if not os.environ.get("CAIRN_DB"):
    os.environ["CAIRN_DB"] = str(ROOT / "cairn.db")


async def orient(prompt: str) -> str | None:
    from cairn._engine_registry import get_engine
    from cairn.models.graph_types import NodeStatus
    from cairn.pipeline.renderer import ViewType, render_structured_summary

    engine = get_engine()

    if engine.graph.node_count() == 0:
        return None

    results = await engine.search_nodes(prompt, k=5)
    active = [(n, s) for n, s in results if n.status == NodeStatus.ACTIVE]
    if not active:
        return None

    focus_ids = [n.id for n, _ in active]
    summary = render_structured_summary(
        engine.graph,
        ViewType.ORIENT,
        focus_node_ids=focus_ids,
        topic=prompt[:100],
    )

    if not summary or "No relevant" in summary:
        return None

    return summary


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    prompt = data.get("prompt", "").strip()
    if not prompt:
        sys.exit(0)

    try:
        summary = asyncio.run(orient(prompt))
    except Exception:
        sys.exit(0)

    if summary:
        print(json.dumps({"additionalContext": summary}))


if __name__ == "__main__":
    main()
