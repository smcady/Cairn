"""Graph-to-narrative rendering with multiple view types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path

from anthropic import AsyncAnthropic

from cairn.models.graph_types import (
    EdgeType,
    IdeaGraph,
    NodeStatus,
    NodeType,
)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "renderer.txt"


class ViewType(str, Enum):
    CURRENT_STATE = "current_state"
    EVOLUTION = "evolution"
    DISAGREEMENT_MAP = "disagreement_map"
    COVERAGE_REPORT = "coverage_report"
    DECISION_LOG = "decision_log"
    ORIENT = "orient"


def _build_current_state_summary(graph: IdeaGraph, workspace_id: str | None = None) -> str:
    """Traverse graph and produce structured data for current state view."""
    sections = []

    def _get_nodes(node_type: NodeType | None = None) -> list:
        if workspace_id is not None:
            nodes = graph.get_nodes_by_workspace(workspace_id)
            if node_type is not None:
                nodes = [n for n in nodes if n.type == node_type]
            return nodes
        return graph.get_all_nodes(node_type)

    # Key propositions
    propositions = [n for n in _get_nodes(NodeType.PROPOSITION) if n.status == NodeStatus.ACTIVE]
    if propositions:
        lines = ["## Active Propositions"]
        for p in sorted(propositions, key=lambda x: x.confidence, reverse=True):
            support_count = sum(
                1 for _, _, e in graph.get_edges_for_node(p.id, "in")
                if e.type == EdgeType.SUPPORTS
            )
            contradiction_count = sum(
                1 for _, _, e in graph.get_edges_for_node(p.id, "in")
                if e.type == EdgeType.CONTRADICTS
            )
            lines.append(
                f"- [{p.id}] (confidence: {p.confidence:.1f}, "
                f"support: {support_count}, challenges: {contradiction_count}, "
                f"depth: {p.depth_of_exploration}) {p.text}"
            )
        sections.append("\n".join(lines))

    # Syntheses
    syntheses = [n for n in _get_nodes(NodeType.SYNTHESIS) if n.status == NodeStatus.ACTIVE]
    if syntheses:
        lines = ["## Syntheses"]
        for s in syntheses:
            constituents = [
                v for _, v, e in graph.get_edges_for_node(s.id, "out")
                if e.type == EdgeType.SYNTHESIZES
            ]
            constituent_texts = []
            for cid in constituents:
                cn = graph.get_node(cid)
                if cn:
                    constituent_texts.append(f"[{cid}]")
            lines.append(f"- [{s.id}] {s.text} (unifies: {', '.join(constituent_texts)})")
        sections.append("\n".join(lines))

    # Open tensions
    tensions = [n for n in _get_nodes(NodeType.TENSION) if n.status == NodeStatus.ACTIVE]
    if tensions:
        lines = ["## Unresolved Tensions"]
        for t in tensions:
            between_ids = [
                v for _, v, e in graph.get_edges_for_node(t.id, "out")
                if e.type == EdgeType.BETWEEN
            ]
            between_texts = []
            for bid in between_ids:
                bn = graph.get_node(bid)
                if bn:
                    between_texts.append(f"[{bid}] {bn.text}")
            lines.append(f"- [{t.id}] {t.text}")
            for bt in between_texts:
                lines.append(f"  Between: {bt}")
        sections.append("\n".join(lines))

    # Open questions
    questions = [n for n in _get_nodes(NodeType.QUESTION) if n.status == NodeStatus.ACTIVE]
    if questions:
        lines = ["## Open Questions"]
        for q in questions:
            related = [
                v for _, v, e in graph.get_edges_for_node(q.id, "out")
                if e.type == EdgeType.QUESTIONS
            ]
            related_str = f" (regarding: {', '.join(f'[{r}]' for r in related)})" if related else ""
            lines.append(f"- [{q.id}] {q.text}{related_str}")
        sections.append("\n".join(lines))

    # Active frames
    frames = [n for n in _get_nodes(NodeType.FRAME) if n.status == NodeStatus.ACTIVE]
    if frames:
        lines = ["## Active Frames/Perspectives"]
        for f in frames:
            lines.append(f"- [{f.id}] {f.text}")
        sections.append("\n".join(lines))

    # Unexplored territories
    territories = [n for n in _get_nodes(NodeType.TERRITORY) if n.status == NodeStatus.ACTIVE]
    if territories:
        lines = ["## Unexplored Territories"]
        for t in territories:
            lines.append(f"- [{t.id}] {t.text}")
        sections.append("\n".join(lines))

    # Summary stats
    all_nodes = _get_nodes()
    active_count = sum(1 for n in all_nodes if n.status == NodeStatus.ACTIVE)
    parked_count = sum(1 for n in all_nodes if n.status == NodeStatus.PARKED)
    resolved_count = sum(1 for n in all_nodes if n.status == NodeStatus.RESOLVED)
    stats = (
        f"## Graph Stats\n"
        f"- Total nodes: {graph.node_count()} (active: {active_count}, "
        f"resolved: {resolved_count}, parked: {parked_count})\n"
        f"- Total edges: {graph.edge_count()}"
    )
    sections.append(stats)

    return "\n\n".join(sections) if sections else "The idea space is empty."


def _build_coverage_report(graph: IdeaGraph) -> str:
    """Identify what's been explored deeply vs. neglected."""
    lines = ["## Coverage Report"]

    propositions = graph.get_all_nodes(NodeType.PROPOSITION)
    if propositions:
        deep = [p for p in propositions if p.depth_of_exploration >= 3]
        shallow = [p for p in propositions if p.depth_of_exploration == 0 and p.status == NodeStatus.ACTIVE]

        if deep:
            lines.append("\n### Deeply Explored")
            for p in deep:
                lines.append(f"- [{p.id}] (depth: {p.depth_of_exploration}) {p.text}")

        if shallow:
            lines.append("\n### Barely Touched")
            for p in shallow:
                lines.append(f"- [{p.id}] {p.text}")

    # Territories are by definition unexplored
    territories = [n for n in graph.get_all_nodes(NodeType.TERRITORY) if n.status == NodeStatus.ACTIVE]
    if territories:
        lines.append("\n### Flagged but Unexplored Territories")
        for t in territories:
            lines.append(f"- [{t.id}] {t.text}")

    # Open questions that nothing resolves
    open_questions = [n for n in graph.get_all_nodes(NodeType.QUESTION) if n.status == NodeStatus.ACTIVE]
    if open_questions:
        lines.append(f"\n### Open Questions: {len(open_questions)}")
        for q in open_questions:
            lines.append(f"- [{q.id}] {q.text}")

    return "\n".join(lines)


def _build_disagreement_map(graph: IdeaGraph) -> str:
    """Map where sources/propositions diverge."""
    lines = ["## Disagreement Map"]

    # Find all CONTRADICTS edges
    all_nodes = graph.get_all_nodes()
    contested = {}
    for node in all_nodes:
        for u, _v, edge in graph.get_edges_for_node(node.id, "in"):
            if edge.type == EdgeType.CONTRADICTS:
                if node.id not in contested:
                    contested[node.id] = {"node": node, "objections": []}
                obj = graph.get_node(u)
                if obj:
                    contested[node.id]["objections"].append(obj)

    if contested:
        for nid, data in contested.items():
            node = data["node"]
            lines.append(f"\n### [{nid}] {node.text} (confidence: {node.confidence:.1f})")
            lines.append("Challenged by:")
            for obj in data["objections"]:
                lines.append(f"  - [{obj.id}] {obj.text}")
    else:
        lines.append("No disagreements identified yet.")

    # Also include active tensions
    tensions = [n for n in graph.get_all_nodes(NodeType.TENSION) if n.status == NodeStatus.ACTIVE]
    if tensions:
        lines.append("\n### Named Tensions")
        for t in tensions:
            lines.append(f"- [{t.id}] {t.text}")

    return "\n".join(lines)


def _build_decision_log(graph: IdeaGraph) -> str:
    """What's been resolved or parked, and why."""
    lines = ["## Decision Log"]

    resolved_questions = [n for n in graph.get_all_nodes(NodeType.QUESTION) if n.status == NodeStatus.RESOLVED]
    if resolved_questions:
        lines.append("\n### Resolved Questions")
        for q in resolved_questions:
            # Find what resolved it
            resolvers = []
            for u, _v, e in graph.get_edges_for_node(q.id, "in"):
                if e.type == EdgeType.RESOLVES:
                    resolver = graph.get_node(u)
                    if resolver:
                        resolvers.append(resolver)
            lines.append(f"- [{q.id}] {q.text}")
            for r in resolvers:
                lines.append(f"  Resolved by: [{r.id}] {r.text}")

    parked = graph.get_nodes_by_status(NodeStatus.PARKED)
    if parked:
        lines.append("\n### Parked / Abandoned")
        for p in parked:
            lines.append(f"- [{p.id}] ({p.type.value}) {p.text}")

    superseded = graph.get_nodes_by_status(NodeStatus.SUPERSEDED)
    if superseded:
        lines.append("\n### Superseded")
        for s in superseded:
            lines.append(f"- [{s.id}] ({s.type.value}) {s.text}")

    if len(lines) == 1:
        lines.append("No decisions recorded yet.")

    return "\n".join(lines)


def _build_evolution(graph: IdeaGraph, node_id: str | None = None) -> str:
    """How a specific position or the whole space changed over time."""
    if node_id:
        node = graph.get_node(node_id)
        if not node:
            return f"Node {node_id} not found."

        lines = [f"## Evolution of [{node_id}]: {node.text}"]
        if node.version_history:
            lines.append("\n### Version History")
            for i, version in enumerate(node.version_history):
                lines.append(f"  v{i}: {version}")
            lines.append(f"  current: {node.text}")

        lines.append(f"\nConfidence: {node.confidence:.1f}")
        lines.append(f"Depth of exploration: {node.depth_of_exploration}")
        lines.append(f"Status: {node.status.value}")

        # Show related edges
        edges = graph.get_edges_for_node(node_id)
        if edges:
            lines.append("\n### Relationships")
            for u, v, e in edges:
                other_id = v if u == node_id else u
                other = graph.get_node(other_id)
                direction = "→" if u == node_id else "←"
                other_text = other.text[:60] if other else "unknown"
                lines.append(f"  {direction} {e.type.value} [{other_id}] {other_text}")

        return "\n".join(lines)
    else:
        return _build_current_state_summary(graph)


def _build_orient_summary(graph: IdeaGraph, focus_node_ids: list[str], topic: str = "") -> str:
    """Compact topic-scoped orientation view for agents entering new territory."""
    if not focus_node_ids:
        return "No relevant context found in reasoning graph."

    focus_id_set = set(focus_node_ids)

    # Identify contested nodes: propositions with CONTRADICTS edges (even if the
    # challenger is outside the focus set), and TENSION nodes (inherently contested).
    contested_ids: set[str] = set()
    for nid in focus_node_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        if node.type == NodeType.TENSION and node.status == NodeStatus.ACTIVE:
            contested_ids.add(nid)
            # Also mark propositions this tension sits between
            for _u, v, e in graph.get_edges_for_node(nid, "out"):
                if e.type == EdgeType.BETWEEN and v in focus_id_set:
                    contested_ids.add(v)
        for u, v, e in graph.get_edges_for_node(nid, "both"):
            if e.type == EdgeType.CONTRADICTS:
                # Mark as contested even if the challenger isn't in the focus set
                contested_ids.add(nid)

    settled: list[GraphNode] = []
    contested: list[GraphNode] = []
    open_qs: list[GraphNode] = []
    resolved_qs: list[GraphNode] = []

    for nid in focus_node_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        if node.type == NodeType.QUESTION:
            if node.status == NodeStatus.ACTIVE:
                open_qs.append(node)
            else:
                resolved_qs.append(node)
        elif node.type == NodeType.TENSION:
            if node.status == NodeStatus.ACTIVE:
                contested.append(node)
        elif node.type == NodeType.PROPOSITION and node.status == NodeStatus.ACTIVE:
            if nid in contested_ids:
                contested.append(node)
            else:
                settled.append(node)

    def _trunc(text: str, max_len: int = 120) -> str:
        return text if len(text) <= max_len else text[:max_len] + "…"

    n = sum(1 for nid in focus_node_ids if graph.get_node(nid) is not None)
    topic_label = f'"{topic}"' if topic else "topic"
    lines = [f"## Orientation: {topic_label}", f"Relevant context from reasoning graph ({n} nodes):"]

    if settled:
        lines.append("\nSETTLED")
        for node in settled:
            lines.append(f"• {_trunc(node.text)}")

    if contested:
        lines.append("\nCONTESTED")
        for node in contested:
            lines.append(f"• {_trunc(node.text)}")

    if open_qs:
        lines.append("\nOPEN QUESTIONS")
        for node in open_qs:
            lines.append(f"• ? {_trunc(node.text)}")

    if resolved_qs:
        lines.append("\nRECENTLY RESOLVED")
        for node in resolved_qs:
            lines.append(f"• {_trunc(node.text)} (resolved)")

    if not (settled or contested or open_qs or resolved_qs):
        return "No relevant context found in reasoning graph."

    return "\n".join(lines)


VIEW_BUILDERS = {
    ViewType.CURRENT_STATE: lambda g, **kw: _build_current_state_summary(g, workspace_id=kw.get("workspace_id")),
    ViewType.EVOLUTION: lambda g, **kw: _build_evolution(g, kw.get("node_id")),
    ViewType.DISAGREEMENT_MAP: lambda g, **_: _build_disagreement_map(g),
    ViewType.COVERAGE_REPORT: lambda g, **_: _build_coverage_report(g),
    ViewType.DECISION_LOG: lambda g, **_: _build_decision_log(g),
    ViewType.ORIENT: lambda g, **kw: _build_orient_summary(
        g, kw.get("focus_node_ids", []), kw.get("topic", "")
    ),
}


def render_structured_summary(
    graph: IdeaGraph,
    view_type: ViewType = ViewType.CURRENT_STATE,
    workspace_id: str | None = None,
    **kwargs,
) -> str:
    """Render a structured text summary of the graph via traversal only (no LLM call).

    This is used in the pipeline critical path to avoid an extra LLM round-trip.
    The evaluator and responder can consume structured data directly.
    """
    if graph.node_count() == 0:
        return "The idea space is empty. No propositions, questions, or other elements have been introduced yet."

    builder = VIEW_BUILDERS[view_type]
    return builder(graph, workspace_id=workspace_id, **kwargs)


async def render_narrative(
    client: AsyncAnthropic,
    graph: IdeaGraph,
    view_type: ViewType = ViewType.CURRENT_STATE,
    model: str = "claude-sonnet-4-5-20250929",
    **kwargs,
) -> str:
    """Render a natural language narrative of the graph state.

    Args:
        client: Anthropic async client
        graph: The idea graph
        view_type: Which view to render
        model: Model to use
        **kwargs: Additional args (e.g., node_id for evolution view)

    Returns:
        Natural language narrative
    """
    # Build structured summary from graph traversal
    builder = VIEW_BUILDERS[view_type]
    structured_summary = builder(graph, **kwargs)

    # If the graph is empty or very small, return the structured summary directly
    if graph.node_count() == 0:
        return "The idea space is empty. No propositions, questions, or other elements have been introduced yet."

    if graph.node_count() <= 3:
        return structured_summary

    system_prompt = PROMPT_PATH.read_text()

    user_message = f"""VIEW TYPE: {view_type.value}

STRUCTURED GRAPH DATA:
{structured_summary}

Convert this structured graph data into a clear, readable narrative. Maintain all the information but present it as prose that a human collaborator would find useful for understanding where the exploration stands."""

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


async def stream_narrative(
    client: AsyncAnthropic,
    graph: IdeaGraph,
    view_type: ViewType = ViewType.CURRENT_STATE,
    model: str = "claude-sonnet-4-5-20250929",
    **kwargs,
) -> AsyncIterator[str]:
    """Stream a natural language narrative of the graph state.

    Yields text deltas as they arrive. For small/empty graphs, yields the
    full result in one chunk (no LLM call needed).
    """
    if graph.node_count() == 0:
        yield "The idea space is empty. No propositions, questions, or other elements have been introduced yet."
        return

    builder = VIEW_BUILDERS[view_type]
    structured_summary = builder(graph, **kwargs)

    if graph.node_count() <= 3:
        yield structured_summary
        return

    system_prompt = PROMPT_PATH.read_text()

    user_message = f"""VIEW TYPE: {view_type.value}

STRUCTURED GRAPH DATA:
{structured_summary}

Convert this structured graph data into a clear, readable narrative. Maintain all the information but present it as prose that a human collaborator would find useful for understanding where the exploration stands."""

    async with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
