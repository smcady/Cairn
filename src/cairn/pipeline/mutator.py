"""Deterministic mutation rules — no LLM calls.

Given a classified event and the current graph state, applies the
corresponding graph operations and returns the list of created/modified node IDs.
"""

from __future__ import annotations

from cairn.models.events import (
    AbandonmentPayload,
    ConnectionPayload,
    ContradictionPayload,
    Event,
    EventType,
    NewPropositionPayload,
    NewQuestionPayload,
    QuestionResolvedPayload,
    ReframePayload,
    RefinementPayload,
    SupportPayload,
    SynthesisPayload,
    TensionIdentifiedPayload,
    TerritoryIdentifiedPayload,
)
from cairn.models.graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    IdeaGraph,
    NodeStatus,
    NodeType,
)


def _stable_id(assigned: str) -> str:
    """Return the pre-assigned node ID from the event payload."""
    return assigned


class MutationResult:
    """Tracks what changed during a mutation."""

    def __init__(self) -> None:
        self.created_node_ids: list[str] = []
        self.modified_node_ids: list[str] = []
        self.created_edges: list[tuple[str, str, EdgeType]] = []


def apply_event(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    """Dispatch an event to its mutation rule and return what changed."""
    handlers = {
        EventType.NEW_PROPOSITION: _handle_new_proposition,
        EventType.SUPPORT: _handle_support,
        EventType.CONTRADICTION: _handle_contradiction,
        EventType.REFINEMENT: _handle_refinement,
        EventType.NEW_QUESTION: _handle_new_question,
        EventType.QUESTION_RESOLVED: _handle_question_resolved,
        EventType.CONNECTION: _handle_connection,
        EventType.TENSION_IDENTIFIED: _handle_tension_identified,
        EventType.TERRITORY_IDENTIFIED: _handle_territory_identified,
        EventType.REFRAME: _handle_reframe,
        EventType.SYNTHESIS: _handle_synthesis,
        EventType.ABANDONMENT: _handle_abandonment,
    }
    handler = handlers[event.event_type]
    return handler(graph, event, workspace_id=workspace_id)


def _handle_new_proposition(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = NewPropositionPayload.model_validate(event.payload)

    node = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.PROPOSITION,
        text=payload.text,
        confidence=0.5,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(node)
    result.created_node_ids.append(node.id)

    for ref_id in payload.related_node_ids:
        if graph.get_node(ref_id) is not None:
            edge = GraphEdge(type=EdgeType.RELATES_TO, timestamp=event.timestamp)
            graph.add_edge(node.id, ref_id, edge)
            result.created_edges.append((node.id, ref_id, EdgeType.RELATES_TO))

    return result


def _handle_support(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = SupportPayload.model_validate(event.payload)

    target = graph.get_node(payload.target_node_id)
    if target is None:
        return result

    evidence = GraphNode(
        id=_stable_id(payload.evidence_node_id),
        type=NodeType.EVIDENCE,
        text=payload.evidence_text,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(evidence)
    result.created_node_ids.append(evidence.id)

    edge = GraphEdge(type=EdgeType.SUPPORTS, timestamp=event.timestamp)
    graph.add_edge(evidence.id, payload.target_node_id, edge)
    result.created_edges.append((evidence.id, payload.target_node_id, EdgeType.SUPPORTS))

    new_confidence = min(0.9, target.confidence + 0.1)
    graph.update_node(payload.target_node_id, confidence=new_confidence)
    result.modified_node_ids.append(payload.target_node_id)

    return result


def _handle_contradiction(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = ContradictionPayload.model_validate(event.payload)

    target = graph.get_node(payload.target_node_id)
    if target is None:
        return result

    objection = GraphNode(
        id=_stable_id(payload.objection_node_id),
        type=NodeType.OBJECTION,
        text=payload.objection_text,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(objection)
    result.created_node_ids.append(objection.id)

    edge = GraphEdge(type=EdgeType.CONTRADICTS, timestamp=event.timestamp)
    graph.add_edge(objection.id, payload.target_node_id, edge)
    result.created_edges.append((objection.id, payload.target_node_id, EdgeType.CONTRADICTS))

    new_confidence = max(0.1, target.confidence - 0.1)
    graph.update_node(payload.target_node_id, confidence=new_confidence)
    result.modified_node_ids.append(payload.target_node_id)

    # Auto-generate tension if confidence drops below 0.3
    if new_confidence < 0.3:
        strongest_objection = _find_strongest_objection(graph, payload.target_node_id)
        if strongest_objection:
            tension = GraphNode(
                type=NodeType.TENSION,
                text=f"Tension between '{target.text}' and '{strongest_objection.text}'",
                status=NodeStatus.ACTIVE,
                timestamp=event.timestamp,
                workspace_id=workspace_id,
            )
            graph.add_node(tension)
            result.created_node_ids.append(tension.id)

            for nid in [payload.target_node_id, strongest_objection.id]:
                e = GraphEdge(type=EdgeType.BETWEEN, timestamp=event.timestamp)
                graph.add_edge(tension.id, nid, e)
                result.created_edges.append((tension.id, nid, EdgeType.BETWEEN))

    return result


def _find_strongest_objection(graph: IdeaGraph, target_id: str) -> GraphNode | None:
    """Find the objection node with the strongest CONTRADICTS edge to target."""
    best: GraphNode | None = None
    best_strength = -1.0
    for u, _v, edge in graph.get_edges_for_node(target_id, direction="in"):
        if edge.type == EdgeType.CONTRADICTS:
            node = graph.get_node(u)
            if node and node.type == NodeType.OBJECTION and edge.strength > best_strength:
                best = node
                best_strength = edge.strength
    return best


def _handle_refinement(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = RefinementPayload.model_validate(event.payload)

    target = graph.get_node(payload.target_node_id)
    if target is None:
        return result

    # Preserve version history
    version_history = list(target.version_history)
    version_history.append(target.text)

    graph.update_node(
        payload.target_node_id,
        text=payload.new_text,
        version_history=version_history,
        depth_of_exploration=target.depth_of_exploration + 1,
    )
    result.modified_node_ids.append(payload.target_node_id)

    return result


def _handle_new_question(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = NewQuestionPayload.model_validate(event.payload)

    node = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.QUESTION,
        text=payload.text,
        source=payload.source,
        status=NodeStatus.ACTIVE,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(node)
    result.created_node_ids.append(node.id)

    for ref_id in payload.related_node_ids:
        if graph.get_node(ref_id) is not None:
            edge = GraphEdge(type=EdgeType.QUESTIONS, timestamp=event.timestamp)
            graph.add_edge(node.id, ref_id, edge)
            result.created_edges.append((node.id, ref_id, EdgeType.QUESTIONS))

    return result


def _handle_question_resolved(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = QuestionResolvedPayload.model_validate(event.payload)

    question = graph.get_node(payload.question_node_id)
    if question is None:
        return result

    graph.update_node(payload.question_node_id, status=NodeStatus.RESOLVED)
    result.modified_node_ids.append(payload.question_node_id)

    # Create a proposition for the resolution
    resolution = GraphNode(
        id=_stable_id(payload.resolution_node_id),
        type=NodeType.PROPOSITION,
        text=payload.resolution_text,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(resolution)
    result.created_node_ids.append(resolution.id)

    edge = GraphEdge(type=EdgeType.RESOLVES, timestamp=event.timestamp)
    graph.add_edge(resolution.id, payload.question_node_id, edge)
    result.created_edges.append((resolution.id, payload.question_node_id, EdgeType.RESOLVES))

    return result


def _handle_connection(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = ConnectionPayload.model_validate(event.payload)

    if graph.get_node(payload.source_node_id) is None or graph.get_node(payload.target_node_id) is None:
        return result

    edge = GraphEdge(
        type=EdgeType.RELATES_TO,
        basis=payload.basis,
        timestamp=event.timestamp,
    )
    graph.add_edge(payload.source_node_id, payload.target_node_id, edge)
    result.created_edges.append((payload.source_node_id, payload.target_node_id, EdgeType.RELATES_TO))

    return result


def _handle_tension_identified(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = TensionIdentifiedPayload.model_validate(event.payload)

    tension = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.TENSION,
        text=payload.description,
        source=payload.source,
        status=NodeStatus.ACTIVE,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(tension)
    result.created_node_ids.append(tension.id)

    for nid in payload.node_ids:
        if graph.get_node(nid) is not None:
            edge = GraphEdge(type=EdgeType.BETWEEN, timestamp=event.timestamp)
            graph.add_edge(tension.id, nid, edge)
            result.created_edges.append((tension.id, nid, EdgeType.BETWEEN))

    return result


def _handle_territory_identified(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = TerritoryIdentifiedPayload.model_validate(event.payload)

    territory = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.TERRITORY,
        text=payload.text,
        source=payload.source,
        status=NodeStatus.ACTIVE,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(territory)
    result.created_node_ids.append(territory.id)

    for ref_id in payload.adjacent_node_ids:
        if graph.get_node(ref_id) is not None:
            edge = GraphEdge(type=EdgeType.ADJACENT_TO, timestamp=event.timestamp)
            graph.add_edge(territory.id, ref_id, edge)
            result.created_edges.append((territory.id, ref_id, EdgeType.ADJACENT_TO))

    return result


def _handle_reframe(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = ReframePayload.model_validate(event.payload)

    frame = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.FRAME,
        text=payload.text,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(frame)
    result.created_node_ids.append(frame.id)

    for nid in payload.affected_node_ids:
        if graph.get_node(nid) is not None:
            edge = GraphEdge(type=EdgeType.REFRAMES, timestamp=event.timestamp)
            graph.add_edge(frame.id, nid, edge)
            result.created_edges.append((frame.id, nid, EdgeType.REFRAMES))

    # Mark prior frames that reframed the same nodes as superseded
    for nid in payload.affected_node_ids:
        for u, _v, e in graph.get_edges_for_node(nid, direction="in"):
            if e.type == EdgeType.REFRAMES and u != frame.id:
                prior = graph.get_node(u)
                if prior and prior.type == NodeType.FRAME and prior.status == NodeStatus.ACTIVE:
                    graph.update_node(u, status=NodeStatus.SUPERSEDED)
                    result.modified_node_ids.append(u)

    return result


def _handle_synthesis(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = SynthesisPayload.model_validate(event.payload)

    synthesis = GraphNode(
        id=_stable_id(payload.node_id),
        type=NodeType.SYNTHESIS,
        text=payload.text,
        source=payload.source,
        timestamp=event.timestamp,
        workspace_id=workspace_id,
    )
    graph.add_node(synthesis)
    result.created_node_ids.append(synthesis.id)

    for nid in payload.constituent_node_ids:
        node = graph.get_node(nid)
        if node is not None:
            edge = GraphEdge(type=EdgeType.SYNTHESIZES, timestamp=event.timestamp)
            graph.add_edge(synthesis.id, nid, edge)
            result.created_edges.append((synthesis.id, nid, EdgeType.SYNTHESIZES))

            if payload.supersedes_constituents and node.status == NodeStatus.ACTIVE:
                graph.update_node(nid, status=NodeStatus.SUPERSEDED)
                result.modified_node_ids.append(nid)

    return result


def _handle_abandonment(graph: IdeaGraph, event: Event, workspace_id: str = "") -> MutationResult:
    result = MutationResult()
    payload = AbandonmentPayload.model_validate(event.payload)

    target = graph.get_node(payload.target_node_id)
    if target is None:
        return result

    graph.update_node(payload.target_node_id, status=NodeStatus.PARKED)
    result.modified_node_ids.append(payload.target_node_id)

    return result


def replay_events(graph: IdeaGraph, events: list[Event]) -> None:
    """Replay a list of events to rebuild graph state from scratch."""
    graph.clear()
    for event in events:
        apply_event(graph, event, workspace_id=event.workspace_id)
