"""Memory engine: two-stage classify → resolve → mutate pipeline for the reasoning graph."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from cairn.models.events import Event, EventLog, EventType, validate_event_payload
from cairn.models.graph_types import GraphNode, IdeaGraph, NodeType
from cairn.pipeline.classifier import ClassifiedResult, classify_exchange
from cairn.pipeline.mutator import apply_event, replay_events
from cairn.pipeline.resolver import resolve_classified_event
from cairn.utils.metrics import SessionMetrics
from cairn.utils.vector_index import VectorIndex


@dataclass
class DroppedEvent:
    """Records why a classified event was dropped during the resolve or validate stage."""

    event_type: str
    reason: str
    unresolved_description: str = ""
    resolution_score: float | None = None


class IngestResult:
    def __init__(self) -> None:
        self.applied_events: list[Event] = []
        self.dropped_events: list[DroppedEvent] = []

    def _dropped_as_tuples(self) -> list[tuple[str, str]]:
        """Backward-compatible view: list of (event_type, reason) tuples."""
        return [(d.event_type, d.reason) for d in self.dropped_events]


class MemoryEngine:
    """Classify → resolve → mutate pipeline and graph state."""

    def __init__(
        self,
        event_log: EventLog,
        graph: IdeaGraph | None = None,
        vector_index: VectorIndex | None = None,
        classifier_model: str = "claude-sonnet-4-5-20250929",
        resolution_threshold: float = 0.82,
    ) -> None:
        self.event_log = event_log
        self.graph = graph or IdeaGraph()
        self.vector_index = vector_index
        self.client = AsyncAnthropic()
        self.session_id = uuid.uuid4().hex[:8]
        self.turn_number = 0
        self.classifier_model = classifier_model
        self.resolution_threshold = resolution_threshold
        self.metrics = SessionMetrics()

    def rebuild_from_log(self) -> None:
        """Replay all stored events to rebuild graph state."""
        events = self.event_log.get_all()
        replay_events(self.graph, events)
        if events:
            self.turn_number = max(e.turn_number or 0 for e in events)

    async def ingest(self, text: str, source: str = "external") -> IngestResult:
        """Two-stage classify → resolve → mutate pipeline.

        Stage 1: LLM identifies event types and describes node references in plain text.
        Stage 2: Vector search resolves text descriptions to actual graph node IDs.
        """
        result = IngestResult()
        self.turn_number += 1

        # Stage 1: classify
        classified_events = await classify_exchange(
            client=self.client,
            exchange_text=text,
            graph=self.graph,
            source=source,
            model=self.classifier_model,
            metrics=self.metrics,
        )

        # Stage 2: resolve + mutate
        for ce in classified_events:
            # Resolve node descriptions to IDs (or fall back to unresolved payload for
            # events that don't reference existing nodes)
            if self.vector_index is not None:
                resolved, drop_info = await resolve_classified_event(
                    ce, self.graph, self.vector_index, self.resolution_threshold
                )
            else:
                # No vector index: only events with no required node refs can proceed
                resolved = _resolve_without_index(ce)
                drop_info = None

            if resolved is None:
                unresolved_desc = drop_info.unresolved_description if drop_info else ""
                best_score = drop_info.resolution_score if drop_info else None
                result.dropped_events.append(
                    DroppedEvent(
                        event_type=ce.event_type.value,
                        reason="node reference could not be resolved",
                        unresolved_description=unresolved_desc,
                        resolution_score=best_score,
                    )
                )
                continue

            validated_payload, error = validate_event_payload(resolved.event_type, resolved.payload)
            if validated_payload is None:
                result.dropped_events.append(
                    DroppedEvent(event_type=ce.event_type.value, reason=error)
                )
                continue

            # Pre-assign stable node IDs so replay produces identical graph structure
            validated_payload = _inject_node_ids(resolved.event_type, validated_payload)

            event = Event(
                event_type=resolved.event_type,
                payload=validated_payload,
                turn_number=self.turn_number,
                session_id=self.session_id,
            )
            self.event_log.append(event)
            mutation = apply_event(self.graph, event)
            result.applied_events.append(event)

            # Index new and modified nodes
            if self.vector_index is not None:
                for node_id in mutation.created_node_ids + mutation.modified_node_ids:
                    node = self.graph.get_node(node_id)
                    if node is not None:
                        await self.vector_index.add(node_id, node.text)

        return result

    async def search_nodes(
        self, query: str, k: int = 10
    ) -> list[tuple[GraphNode, float]]:
        """Semantic search over indexed graph nodes. Returns (node, score) pairs."""
        if self.vector_index is None or len(self.vector_index) == 0:
            return []
        results = await self.vector_index.search(query, k=k)
        out = []
        for node_id, score in results:
            node = self.graph.get_node(node_id)
            if node is not None:
                out.append((node, score))
        return out

    def get_stats(self) -> dict:
        """Return summary stats about the current graph."""
        all_nodes = self.graph.get_all_nodes()
        stats = {
            "total_nodes": self.graph.node_count(),
            "total_edges": self.graph.edge_count(),
            "active": sum(1 for n in all_nodes if n.status.value == "active"),
            "resolved": sum(1 for n in all_nodes if n.status.value == "resolved"),
            "parked": sum(1 for n in all_nodes if n.status.value == "parked"),
            "propositions": len(self.graph.get_all_nodes(node_type=NodeType.PROPOSITION)),
            "questions": len(self.graph.get_all_nodes(node_type=NodeType.QUESTION)),
            "tensions": len(self.graph.get_all_nodes(node_type=NodeType.TENSION)),
            "turns": self.turn_number,
            "total_events": self.event_log.count(),
        }
        if self.vector_index is not None:
            stats["indexed_nodes"] = len(self.vector_index)
            stats["embed_requests"] = self.vector_index._request_count
            if self.vector_index._max_requests is not None:
                stats["embed_budget"] = self.vector_index._max_requests
            self.metrics.embed_calls = self.vector_index._request_count
        stats["metrics"] = self.metrics.to_dict()
        return stats


# Maps event type to the payload field that should hold the pre-assigned node ID
_NODE_ID_FIELDS: dict[EventType, str] = {
    EventType.NEW_PROPOSITION:      "node_id",
    EventType.NEW_QUESTION:         "node_id",
    EventType.SUPPORT:              "evidence_node_id",
    EventType.CONTRADICTION:        "objection_node_id",
    EventType.QUESTION_RESOLVED:    "resolution_node_id",
    EventType.TENSION_IDENTIFIED:   "node_id",
    EventType.TERRITORY_IDENTIFIED: "node_id",
    EventType.REFRAME:              "node_id",
    EventType.SYNTHESIS:            "node_id",
}


def _inject_node_ids(event_type: EventType, payload: dict) -> dict:
    """Pre-assign a stable node ID for node-creating events before storing to the log.

    Ensures that replay_events() produces the same graph structure as the original run,
    preserving referential integrity for events like QUESTION_RESOLVED that reference
    node IDs created by prior events.
    """
    field = _NODE_ID_FIELDS.get(event_type)
    if field is None:
        return payload
    if not payload.get(field):
        payload = {**payload, field: uuid.uuid4().hex[:12]}
    return payload


def _resolve_without_index(ce) -> ClassifiedResult | None:
    """Fallback when no VectorIndex is available.

    Only events that require no existing node references can proceed.
    All others are dropped.
    """
    from cairn.models.events import EventType

    no_ref_types = {
        EventType.NEW_PROPOSITION,
        EventType.NEW_QUESTION,
        EventType.TERRITORY_IDENTIFIED,
        EventType.REFRAME,
    }
    if ce.event_type not in no_ref_types:
        return None

    payload: dict = {"text": ce.text, "source": ce.source}
    if ce.event_type == EventType.NEW_PROPOSITION:
        payload["related_node_ids"] = []
    elif ce.event_type == EventType.NEW_QUESTION:
        payload["related_node_ids"] = []
    elif ce.event_type == EventType.TERRITORY_IDENTIFIED:
        payload["adjacent_node_ids"] = []
    elif ce.event_type == EventType.REFRAME:
        payload["affected_node_ids"] = []

    return ClassifiedResult(event_type=ce.event_type, payload=payload, reasoning=ce.reasoning)
