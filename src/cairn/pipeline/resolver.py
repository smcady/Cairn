"""Stage 2: resolve text descriptions in ClassifiedEvents to actual graph node IDs.

The classifier (Stage 1) produces ClassifiedEvent objects where node references are
expressed as plain-text descriptions (e.g. "the proposition about AI transforming work"),
not opaque hex IDs. This module resolves those descriptions to real node IDs via vector
search, then produces ClassifiedResult objects with validated payload dicts.

If a required node reference cannot be resolved above the confidence threshold, the whole
event is dropped (returns None) rather than risk creating a bad edge.
Optional references that fail resolution are silently omitted from the payload list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cairn.models.events import ClassifiedEvent, EventType
from cairn.models.graph_types import IdeaGraph, NodeStatus
from cairn.pipeline.classifier import ClassifiedResult
from cairn.utils.vector_index import VectorIndex

DEFAULT_THRESHOLD = 0.82


@dataclass
class ResolveDropInfo:
    """Carries debug info when a required node reference fails to resolve."""

    unresolved_description: str = ""
    resolution_score: float | None = None

# ---------------------------------------------------------------------------
# Per-event-type field mappings
# ---------------------------------------------------------------------------

# Required single ref: (description_field, payload_id_field)
_REQUIRED_SINGLE: dict[EventType, tuple[str, str]] = {
    EventType.SUPPORT:            ("target_node_description", "target_node_id"),
    EventType.CONTRADICTION:      ("target_node_description", "target_node_id"),
    EventType.REFINEMENT:         ("target_node_description", "target_node_id"),
    EventType.ABANDONMENT:        ("target_node_description", "target_node_id"),
    EventType.QUESTION_RESOLVED:  ("question_node_description", "question_node_id"),
}

# Required dual refs for CONNECTION: both must resolve
_CONNECTION_REFS = [
    ("source_node_description", "source_node_id"),
    ("target_node_description", "target_node_id"),
]

# Required list refs: (description_field, payload_id_field, min_count)
_REQUIRED_LIST: dict[EventType, tuple[str, str, int]] = {
    EventType.TENSION_IDENTIFIED: ("node_descriptions",           "node_ids",              2),
    EventType.SYNTHESIS:          ("constituent_node_descriptions","constituent_node_ids",  2),
}

# Optional list refs: (description_field, payload_id_field)
_OPTIONAL_LIST: dict[EventType, tuple[str, str]] = {
    EventType.NEW_PROPOSITION:      ("related_node_descriptions",   "related_node_ids"),
    EventType.NEW_QUESTION:         ("related_node_descriptions",   "related_node_ids"),
    EventType.TERRITORY_IDENTIFIED: ("adjacent_node_descriptions",  "adjacent_node_ids"),
    EventType.REFRAME:              ("affected_node_descriptions",  "affected_node_ids"),
}

# Non-reference payload fields to copy directly from ClassifiedEvent
_PLAIN_FIELDS: dict[EventType, list[str]] = {
    EventType.NEW_PROPOSITION:      ["text", "source"],
    EventType.SUPPORT:              ["evidence_text", "source"],
    EventType.CONTRADICTION:        ["objection_text", "source"],
    EventType.REFINEMENT:           ["new_text", "source"],
    EventType.NEW_QUESTION:         ["text", "source"],
    EventType.QUESTION_RESOLVED:    ["resolution_text", "source"],
    EventType.CONNECTION:           ["basis"],
    EventType.TENSION_IDENTIFIED:   ["description", "source"],
    EventType.TERRITORY_IDENTIFIED: ["text", "source"],
    EventType.REFRAME:              ["text", "source"],
    EventType.SYNTHESIS:            ["text", "source", "supersedes_constituents"],
    EventType.ABANDONMENT:          ["reason"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_node_reference(
    description: str,
    graph: IdeaGraph,
    vector_index: VectorIndex,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str | None, float | None]:
    """Embed a text description and return (best matching active node ID, best score).

    Returns (None, None) when description is empty or index is empty.
    Returns (None, score) when the best candidate is below threshold or not active.
    Returns (node_id, score) on successful resolution.
    """
    if not description.strip():
        return None, None
    results = await vector_index.search(description, k=3)
    if not results:
        return None, None
    top_id, top_score = results[0]
    if top_score < threshold:
        return None, top_score
    node = graph.get_node(top_id)
    if node is None or node.status not in (NodeStatus.ACTIVE,):
        return None, top_score
    return top_id, top_score


async def resolve_classified_event(
    event: ClassifiedEvent,
    graph: IdeaGraph,
    vector_index: VectorIndex,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[ClassifiedResult | None, ResolveDropInfo | None]:
    """Resolve all node description fields to IDs and return (ClassifiedResult, drop_info).

    Returns (None, ResolveDropInfo) if a required reference cannot be resolved.
    Returns (ClassifiedResult, None) on success.
    """
    et = event.event_type
    data = event.model_dump()
    payload: dict[str, Any] = {}

    # 1. Copy plain (non-ref) fields
    for field_name in _PLAIN_FIELDS.get(et, []):
        payload[field_name] = data.get(field_name, "")

    # 2. Required single ref
    if et in _REQUIRED_SINGLE:
        desc_field, id_field = _REQUIRED_SINGLE[et]
        description = data.get(desc_field, "")
        resolved_id, best_score = await resolve_node_reference(description, graph, vector_index, threshold)
        if resolved_id is None:
            return None, ResolveDropInfo(
                unresolved_description=description,
                resolution_score=best_score,
            )
        payload[id_field] = resolved_id

    # 3. CONNECTION: two required refs
    elif et == EventType.CONNECTION:
        for desc_field, id_field in _CONNECTION_REFS:
            description = data.get(desc_field, "")
            resolved_id, best_score = await resolve_node_reference(description, graph, vector_index, threshold)
            if resolved_id is None:
                return None, ResolveDropInfo(
                    unresolved_description=description,
                    resolution_score=best_score,
                )
            payload[id_field] = resolved_id

    # 4. Required list refs (TENSION_IDENTIFIED, SYNTHESIS)
    if et in _REQUIRED_LIST:
        desc_field, id_field, min_count = _REQUIRED_LIST[et]
        descriptions: list[str] = data.get(desc_field, [])
        resolved_ids: list[str] = []
        first_failed_desc: str = ""
        first_failed_score: float | None = None
        for desc in descriptions:
            nid, score = await resolve_node_reference(desc, graph, vector_index, threshold)
            if nid is not None:
                resolved_ids.append(nid)
            elif not first_failed_desc:
                first_failed_desc = desc
                first_failed_score = score
        if len(resolved_ids) < min_count:
            return None, ResolveDropInfo(
                unresolved_description=first_failed_desc,
                resolution_score=first_failed_score,
            )
        payload[id_field] = resolved_ids

    # 5. Optional list refs
    if et in _OPTIONAL_LIST:
        desc_field, id_field = _OPTIONAL_LIST[et]
        descriptions = data.get(desc_field, [])
        resolved_ids = []
        for desc in descriptions:
            nid, _ = await resolve_node_reference(desc, graph, vector_index, threshold)
            if nid is not None:
                resolved_ids.append(nid)
        payload[id_field] = resolved_ids

    return ClassifiedResult(
        event_type=et,
        payload=payload,
        reasoning=event.reasoning,
    ), None
