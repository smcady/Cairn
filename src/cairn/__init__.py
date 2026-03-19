"""Cairn: a persistent reasoning graph for AI conversations."""

from __future__ import annotations

from cairn._engine_registry import get_engine, init
from cairn.memory.engine import MemoryEngine
from cairn.models.events import EventLog
from cairn.models.graph_types import IdeaGraph, NodeStatus
from cairn.pipeline.renderer import ViewType, render_structured_summary
from cairn.utils.vector_index import VectorIndex

__all__ = [
    "MemoryEngine",
    "EventLog",
    "IdeaGraph",
    "VectorIndex",
    "ViewType",
    "get_engine",
    "init",
    "orient",
    "query",
]


async def orient(topic: str, k: int = 10, db_path: str | None = None) -> str:
    """Search the reasoning graph for prior thinking on a topic.

    Returns a structured summary of settled positions, contested claims,
    and open questions related to the topic. Returns an empty string
    if the graph has no relevant context.
    """
    if not topic.strip():
        return ""
    engine = get_engine(db_path)
    if engine.graph.node_count() == 0:
        return ""
    results = await engine.search_nodes(topic, k=k)
    active = [n for n, _score in results if n.status == NodeStatus.ACTIVE]
    if not active:
        return ""
    focus_ids = [n.id for n in active]
    summary = render_structured_summary(
        engine.graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic=topic
    )
    if not summary or "No relevant" in summary:
        return ""
    return summary


def query(view: str, db_path: str | None = None) -> str:
    """Render a structured view of the reasoning graph.

    Views: 'current_state', 'disagreement_map', 'coverage_report', 'decision_log'
    """
    view_map = {
        "current_state": ViewType.CURRENT_STATE,
        "disagreement_map": ViewType.DISAGREEMENT_MAP,
        "coverage_report": ViewType.COVERAGE_REPORT,
        "decision_log": ViewType.DECISION_LOG,
    }
    if view not in view_map:
        raise ValueError(f"Unknown view '{view}'. Valid: {', '.join(view_map)}")
    engine = get_engine(db_path)
    return render_structured_summary(engine.graph, view_type=view_map[view])
