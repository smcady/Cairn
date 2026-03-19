"""Cairn MCP server — exposes the reasoning graph as tools for Claude Code and other MCP clients."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env.local: try cwd first (Claude Code sets cwd via .mcp.json),
# then fall back to the project root (two levels up from this file).
# This ensures API keys load regardless of how the server is launched.
_project_root = Path(__file__).resolve().parent.parent.parent
_env_local = Path.cwd() / ".env.local"
if not _env_local.exists():
    _env_local = _project_root / ".env.local"
load_dotenv(_env_local, override=True)

from mcp.server.fastmcp import FastMCP

from cairn.memory.engine import MemoryEngine
from cairn.models.events import EventLog, EventType
from cairn.models.graph_types import IdeaGraph, NodeStatus
from cairn.pipeline.mutator import apply_event
from cairn.pipeline.renderer import ViewType, render_structured_summary
from cairn.utils.vector_index import VectorIndex

DB_PATH = os.environ.get("CAIRN_DB", "cairn.db")

mcp = FastMCP(
    "cairn",
    instructions="""\
You have access to a persistent reasoning graph that tracks the state of the user's \
thinking across sessions. It records not just what was said, but what was concluded, \
what was rejected, what's still open, and how confident each position is. Conversations \
are automatically captured — you don't need to manually ingest them.

USE THE GRAPH PROACTIVELY:
- When the user asks about a topic you've discussed before, call harness_orient or \
harness_search BEFORE answering to check what's already been established.
- When the user asks "what did we decide about X" or "where did we land on X", call \
harness_query with 'decision_log' to give a precise answer.
- When you notice the conversation shifting to a new domain, call harness_orient to \
check if there's existing context.
- When the user raises a point that may have already been addressed, search the graph \
before re-opening the discussion.

The graph prevents you from re-opening settled questions, losing the thread across \
sessions, or hedging on positions that have already been resolved. Use it.\
""",
)

_engine: MemoryEngine | None = None
_last_event_id: int = 0


def _get_engine() -> MemoryEngine:
    """Get or update the engine, applying only events added since last check."""
    global _engine, _last_event_id
    if _engine is None:
        event_log = EventLog(DB_PATH)
        graph = IdeaGraph()
        vector_index = VectorIndex(DB_PATH)
        _engine = MemoryEngine(event_log=event_log, graph=graph, vector_index=vector_index)
        _engine.rebuild_from_log()
        last = _engine.event_log.get_recent(1)
        _last_event_id = last[0].id if last else 0
    else:
        new_events = _engine.event_log.get_since(_last_event_id)
        if new_events:
            for event in new_events:
                apply_event(_engine.graph, event, workspace_id=event.workspace_id)
            _last_event_id = new_events[-1].id
    return _engine


@mcp.tool(name="harness_status")
def status() -> str:
    """Quick overview of the reasoning graph — how many ideas are tracked, what's settled, what's contested, what's open.

    Call this at the start of a conversation to see if there's existing context, or when the user asks "where are we?" or "what's the state of things?"
    """
    engine = _get_engine()
    stats = engine.get_stats()

    # Health check: surface problems an implementor would otherwise discover too late
    warnings = []
    if stats["total_events"] == 0:
        warnings.append(
            "⚠ No events ingested. The graph is empty.\n"
            "  If you expected captured conversations, check that the Stop hook\n"
            "  is configured in .claude/settings.json and that CAIRN_DB points\n"
            "  to the correct database file."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        warnings.append("⚠ ANTHROPIC_API_KEY is not set. The classifier will fail on ingest.")
    if not os.environ.get("VOYAGE_API_KEY"):
        warnings.append(
            "ℹ VOYAGE_API_KEY is not set. Using local embeddings (fastembed). "
            "Set VOYAGE_API_KEY for higher-quality Voyage AI embeddings."
        )

    summary = render_structured_summary(engine.graph)
    stats_lines = "\n".join(f"  {k}: {v}" for k, v in stats.items())
    parts = [f"## Graph Stats\n{stats_lines}"]
    if warnings:
        parts.append("## Warnings\n" + "\n\n".join(warnings))
    parts.append(summary)
    return "\n\n".join(parts)


@mcp.tool(name="harness_query")
def query(view: str) -> str:
    """Render a specific view of the reasoning graph.

    Views:
    - 'current_state': all active positions, open questions, and live tensions. Use when the user wants to see where things stand overall.
    - 'disagreement_map': positions that have been contradicted or are in tension. Use when the user asks what's unresolved or contested.
    - 'coverage_report': what topics have been explored vs what's thin. Use when the user asks what hasn't been addressed yet.
    - 'decision_log': settled positions and how they got there. Use when the user asks "what did we decide?" or "why did we go with X?"

    view: one of 'current_state', 'disagreement_map', 'coverage_report', 'decision_log'
    """
    view_map = {
        "current_state": ViewType.CURRENT_STATE,
        "disagreement_map": ViewType.DISAGREEMENT_MAP,
        "coverage_report": ViewType.COVERAGE_REPORT,
        "decision_log": ViewType.DECISION_LOG,
    }
    if view not in view_map:
        return f"Unknown view '{view}'. Valid: {', '.join(view_map)}"
    engine = _get_engine()
    return render_structured_summary(engine.graph, view_type=view_map[view])


@mcp.tool(name="harness_ingest")
async def ingest(content: str, source: str = "mcp") -> str:
    """Manually ingest content into the reasoning graph.

    NOTE: In Claude Code, conversations are captured automatically via the Stop hook — you do NOT need to call this for normal conversation content. Use this only for content from outside the conversation: meeting notes, document excerpts, decisions made elsewhere, or information the user pastes in that should be tracked.

    content: text to classify and ingest
    source: label for where this came from (e.g. 'meeting-notes', 'document', 'slack')
    """
    engine = _get_engine()
    result = await engine.ingest(content, source=source)

    if not result.applied_events:
        dropped = len(result.dropped_events)
        return f"No events extracted. {dropped} dropped." if dropped else "No events extracted."

    lines = [f"Ingested {len(result.applied_events)} event(s) from '{source}':"]
    for event in result.applied_events:
        lines.append(f"  - {event.event_type}")
    if result.dropped_events:
        lines.append(f"({len(result.dropped_events)} dropped: unresolved node references or validation errors)")

    # Surface territory signals so the agent knows to re-orient
    territory_events = [
        e for e in result.applied_events
        if e.event_type == EventType.TERRITORY_IDENTIFIED
    ]
    prior_nodes_exist = engine.graph.node_count() > len(result.applied_events)
    if territory_events and prior_nodes_exist:
        for te in territory_events:
            territory_text = te.payload.get("text", "")
            if territory_text:
                lines.append(f'\nNew territory entered: "{territory_text}"')
                lines.append(f'→ call harness_orient("{territory_text}") for relevant context')

    return "\n".join(lines)


@mcp.tool(name="harness_debug")
async def debug(content: str, source: str = "debug") -> str:
    """
    Ingest content and return the full decision trail — for testing and debugging.

    Returns a JSON object with:
    - applied: list of {event_type, payload, reasoning} for each applied event
    - dropped: list of {event_type, reason, unresolved_description, resolution_score}
    - graph_after: {node_count, edge_count, indexed_nodes} snapshot after ingest
    - turn_number: the turn number after this ingest

    content: text to classify and ingest
    source: where this came from (default 'debug')
    """
    engine = _get_engine()
    result = await engine.ingest(content, source=source)

    applied = [
        {
            "event_type": e.event_type.value,
            "payload": e.payload,
            "reasoning": "",  # reasoning not stored on Event; carried by ClassifiedResult
        }
        for e in result.applied_events
    ]

    dropped = [
        {
            "event_type": d.event_type,
            "reason": d.reason,
            "unresolved_description": d.unresolved_description,
            "resolution_score": d.resolution_score,
        }
        for d in result.dropped_events
    ]

    stats = engine.get_stats()
    graph_after = {
        "node_count": stats["total_nodes"],
        "edge_count": stats["total_edges"],
        "indexed_nodes": stats.get("indexed_nodes", 0),
    }

    output = {
        "applied": applied,
        "dropped": dropped,
        "graph_after": graph_after,
        "turn_number": engine.turn_number,
    }
    return json.dumps(output, indent=2)


@mcp.tool(name="harness_search")
async def search(query: str, k: int = 10) -> str:
    """Search the reasoning graph for what's been established about a topic.

    Returns nodes ranked by relevance with their type (PROPOSITION, QUESTION, TENSION, etc.), status (active, superseded, resolved), and confidence. Use this before answering questions about topics the user has discussed in prior sessions — the graph may already have a settled position.

    query: natural language search query
    k: number of results to return (default 10)
    """
    engine = _get_engine()
    results = await engine.search_nodes(query, k=k)

    if not results:
        return "No indexed nodes found. Try ingesting some content first."

    lines = [f"Top {len(results)} results for: '{query}'\n"]
    for node, score in results:
        lines.append(f"[{node.type.value.upper()}] (score: {score:.3f}, status: {node.status.value})")
        lines.append(f"  {node.text}")
        lines.append(f"  id: {node.id}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(name="harness_orient")
async def orient(topic: str, k: int = 10) -> str:
    """Orient yourself on a topic before responding — what's settled, what's contested, what's open.

    Returns a compact summary scoped to the given topic: settled propositions, contested claims, open questions, and recent resolutions. Call this BEFORE answering when:
    - The conversation shifts to a topic discussed in prior sessions
    - The user re-raises something that may already be resolved
    - You want to build on prior thinking rather than starting from scratch

    This is your most important tool for maintaining continuity across sessions.

    topic: the subject to orient around
    k: number of nodes to retrieve via semantic search (default 10)
    """
    if not topic.strip():
        return "Error: topic must be a non-empty string."
    engine = _get_engine()
    if engine.vector_index is None:
        return "Semantic search unavailable (no vector index configured)."
    search_results = await engine.search_nodes(topic, k=k)
    if not search_results:
        return "No relevant context found in reasoning graph."
    focus_nodes = [n for n, _score in search_results if n.status == NodeStatus.ACTIVE]
    if not focus_nodes:
        return "No relevant context found in reasoning graph."
    focus_node_ids = [n.id for n in focus_nodes]
    return render_structured_summary(engine.graph, ViewType.ORIENT, focus_node_ids=focus_node_ids, topic=topic)


@mcp.tool(name="harness_trace")
def trace(node_id: str) -> str:
    """Trace the full history of a specific idea — how it was proposed, challenged, refined, or resolved.

    Use when the user asks "how did we arrive at this?" or "what changed our mind about X?" Returns the chronological chain of events that shaped this node: when it was created, what contradicted it, how it was refined, and what it became. Includes hash-chain integrity verification.

    node_id: the ID of the node to trace (get IDs from harness_search or harness_query results)
    """
    engine = _get_engine()
    node = engine.graph.get_node(node_id)
    if node is None:
        return f"Node '{node_id}' not found in graph."

    events = engine.event_log.get_all()
    relevant = [e for e in events if node_id in json.dumps(e.payload)]

    if not relevant:
        return f"No events found referencing node '{node_id}'."

    lines = [
        f"## Decision trace: {node.text[:100]}",
        f"Type: {node.type.value} | Status: {node.status.value} | ID: {node_id}",
        "",
    ]

    for e in relevant:
        p = e.payload
        detail = (
            p.get("text") or
            p.get("evidence_text") or
            p.get("objection_text") or
            p.get("resolution_text") or
            p.get("new_text") or
            p.get("reason") or
            p.get("description") or
            p.get("basis") or
            ""
        )
        turn = e.turn_number or "?"
        hash_snippet = f" [↩ {e.parent_hash[:8]}]" if e.parent_hash else " [chain origin]"
        lines.append(f"Turn {turn} · {e.event_type.value}{hash_snippet}")
        if detail:
            lines.append(f"  {detail[:120]}")

    ok, chain_msg = engine.event_log.verify_chain()
    lines.append("")
    lines.append(f"Chain integrity: {'✓' if ok else '✗'} {chain_msg}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
