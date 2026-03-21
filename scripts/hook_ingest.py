#!/usr/bin/env python3
"""
Claude Code Stop hook: ingest the latest conversation turn into Cairn.

Reads stdin JSON from Claude Code (contains transcript_path), extracts the last
user message and assistant response, and ingests both into the reasoning graph.
Runs the ingest in the background so it doesn't block the next turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load project .env.local from cwd (hooks run with cwd = project root)
_env_local = Path.cwd() / ".env.local"
if _env_local.exists():
    load_dotenv(_env_local, override=True)

DB_PATH = os.environ.get("CAIRN_DB", "cairn.db")


def extract_last_turn(transcript_path: str) -> tuple[str, str]:
    """Return (user_text, assistant_text) for the most recent turn."""
    entries = []
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    user_text = ""
    assistant_text = ""

    for entry in reversed(entries):
        entry_type = entry.get("type")
        msg = entry.get("message", {})
        content = msg.get("content", "")

        if entry_type == "assistant" and not assistant_text:
            assistant_text = text_from_content(content).strip()
        elif entry_type == "user" and not user_text:
            # Skip hook feedback / system injections — only grab human turns
            role = msg.get("role", "")
            if role == "user":
                user_text = text_from_content(content).strip()

        if user_text and assistant_text:
            break

    return user_text, assistant_text


async def ingest(content: str) -> None:
    from cairn.memory.engine import MemoryEngine
    from cairn.models.events import EventLog
    from cairn.utils.vector_index import VectorIndex

    event_log = EventLog(DB_PATH)
    vector_index = VectorIndex(DB_PATH)
    engine = MemoryEngine.from_cache(event_log=event_log, vector_index=vector_index)
    await engine.ingest(content, source="claude-code-session")
    engine.save_graph_cache()


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    transcript_path = data.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    user_text, assistant_text = extract_last_turn(transcript_path)

    if not user_text and not assistant_text:
        sys.exit(0)

    parts = []
    if user_text:
        parts.append(f"User: {user_text}")
    if assistant_text:
        parts.append(f"Assistant: {assistant_text}")
    content = "\n\n".join(parts)

    asyncio.run(ingest(content))


if __name__ == "__main__":
    main()
