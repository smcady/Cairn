"""Tests for the Stage 2 resolver: description → node ID via mocked VectorIndex."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cairn.models.events import ClassifiedEvent, EventType
from cairn.models.graph_types import GraphNode, IdeaGraph, NodeStatus, NodeType
from cairn.pipeline.classifier import ClassifiedResult
from cairn.pipeline.resolver import DEFAULT_THRESHOLD, resolve_classified_event, resolve_node_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_with(*nodes: GraphNode) -> IdeaGraph:
    g = IdeaGraph()
    for n in nodes:
        g.add_node(n)
    return g


def _active_prop(node_id: str, text: str = "some proposition") -> GraphNode:
    return GraphNode(id=node_id, type=NodeType.PROPOSITION, text=text, status=NodeStatus.ACTIVE)


def _mock_index(results: list[tuple[str, float]]) -> MagicMock:
    """VectorIndex mock whose search() always returns the given results."""
    idx = MagicMock()
    idx.search = AsyncMock(return_value=results)
    return idx


# ---------------------------------------------------------------------------
# resolve_node_reference
# ---------------------------------------------------------------------------

class TestResolveNodeReference:
    async def test_resolves_above_threshold(self):
        node_id = "abc123"
        graph = _graph_with(_active_prop(node_id))
        idx = _mock_index([(node_id, 0.90)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id == node_id
        assert score == pytest.approx(0.90)

    async def test_returns_none_below_threshold(self):
        node_id = "abc123"
        graph = _graph_with(_active_prop(node_id))
        idx = _mock_index([(node_id, 0.70)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None
        assert score == pytest.approx(0.70)

    async def test_returns_none_at_exact_threshold(self):
        """Score must be >= threshold, not just close."""
        node_id = "abc123"
        graph = _graph_with(_active_prop(node_id))
        idx = _mock_index([(node_id, DEFAULT_THRESHOLD - 0.001)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None

    async def test_accepts_score_equal_to_threshold(self):
        node_id = "abc123"
        graph = _graph_with(_active_prop(node_id))
        idx = _mock_index([(node_id, DEFAULT_THRESHOLD)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id == node_id

    async def test_returns_none_for_superseded_node(self):
        node_id = "abc123"
        node = GraphNode(id=node_id, type=NodeType.PROPOSITION, text="x", status=NodeStatus.SUPERSEDED)
        graph = _graph_with(node)
        idx = _mock_index([(node_id, 0.95)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None

    async def test_returns_none_for_parked_node(self):
        node_id = "abc123"
        node = GraphNode(id=node_id, type=NodeType.PROPOSITION, text="x", status=NodeStatus.PARKED)
        graph = _graph_with(node)
        idx = _mock_index([(node_id, 0.95)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None

    async def test_returns_none_when_node_not_in_graph(self):
        graph = IdeaGraph()  # empty
        idx = _mock_index([("ghost_id", 0.99)])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None

    async def test_returns_none_for_empty_description(self):
        graph = _graph_with(_active_prop("n1"))
        idx = _mock_index([("n1", 0.99)])

        resolved_id, score = await resolve_node_reference("  ", graph, idx)
        assert resolved_id is None
        assert score is None
        idx.search.assert_not_called()

    async def test_returns_none_when_search_empty(self):
        graph = _graph_with(_active_prop("n1"))
        idx = _mock_index([])

        resolved_id, score = await resolve_node_reference("some description", graph, idx)
        assert resolved_id is None
        assert score is None

    async def test_custom_threshold(self):
        node_id = "abc123"
        graph = _graph_with(_active_prop(node_id))
        # Score 0.75 passes threshold=0.70 but not the default 0.82
        idx = _mock_index([(node_id, 0.75)])

        resolved_id, score = await resolve_node_reference("desc", graph, idx, threshold=0.70)
        assert resolved_id == node_id


# ---------------------------------------------------------------------------
# resolve_classified_event: NEW_PROPOSITION (no required refs)
# ---------------------------------------------------------------------------

class TestResolveNewProposition:
    async def test_resolves_with_no_related_nodes(self):
        event = ClassifiedEvent(
            event_type=EventType.NEW_PROPOSITION,
            text="AI will transform software development",
            source="user",
        )
        graph = IdeaGraph()
        idx = _mock_index([])

        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert drop_info is None
        assert result.event_type == EventType.NEW_PROPOSITION
        assert result.payload["text"] == "AI will transform software development"
        assert result.payload["source"] == "user"
        assert result.payload["related_node_ids"] == []

    async def test_resolves_optional_related_nodes(self):
        n1 = _active_prop("node1", "existing idea")
        graph = _graph_with(n1)
        idx = _mock_index([("node1", 0.90)])

        event = ClassifiedEvent(
            event_type=EventType.NEW_PROPOSITION,
            text="new idea",
            related_node_descriptions=["existing idea"],
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["related_node_ids"] == ["node1"]

    async def test_silently_drops_unresolved_optional_refs(self):
        graph = IdeaGraph()  # empty — nothing to resolve to
        idx = _mock_index([])

        event = ClassifiedEvent(
            event_type=EventType.NEW_PROPOSITION,
            text="new idea",
            related_node_descriptions=["some unresolvable description"],
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["related_node_ids"] == []


# ---------------------------------------------------------------------------
# resolve_classified_event: SUPPORT (required single ref)
# ---------------------------------------------------------------------------

class TestResolveSupport:
    async def test_resolves_required_target(self):
        n1 = _active_prop("target1", "AI is transformative")
        graph = _graph_with(n1)
        idx = _mock_index([("target1", 0.91)])

        event = ClassifiedEvent(
            event_type=EventType.SUPPORT,
            target_node_description="AI is transformative",
            evidence_text="GPT-4 shows this",
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert drop_info is None
        assert result.payload["target_node_id"] == "target1"
        assert result.payload["evidence_text"] == "GPT-4 shows this"

    async def test_drops_event_when_required_ref_unresolvable(self):
        graph = IdeaGraph()
        idx = _mock_index([])

        event = ClassifiedEvent(
            event_type=EventType.SUPPORT,
            target_node_description="unresolvable target",
            evidence_text="some evidence",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None
        assert drop_info is not None
        assert drop_info.unresolved_description == "unresolvable target"

    async def test_drops_event_when_score_below_threshold(self):
        n1 = _active_prop("target1")
        graph = _graph_with(n1)
        idx = _mock_index([("target1", 0.50)])

        event = ClassifiedEvent(
            event_type=EventType.SUPPORT,
            target_node_description="some target",
            evidence_text="evidence",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None
        assert drop_info is not None
        assert drop_info.resolution_score == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# resolve_classified_event: CONTRADICTION (required single ref)
# ---------------------------------------------------------------------------

class TestResolveContradiction:
    async def test_resolves_required_target(self):
        n1 = _active_prop("target1", "AI is safe")
        graph = _graph_with(n1)
        idx = _mock_index([("target1", 0.88)])

        event = ClassifiedEvent(
            event_type=EventType.CONTRADICTION,
            target_node_description="AI is safe",
            objection_text="AI systems can be biased",
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["target_node_id"] == "target1"
        assert result.payload["objection_text"] == "AI systems can be biased"

    async def test_drops_if_unresolvable(self):
        graph = IdeaGraph()
        idx = _mock_index([])
        event = ClassifiedEvent(
            event_type=EventType.CONTRADICTION,
            target_node_description="nonexistent",
            objection_text="objection",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_classified_event: CONNECTION (two required refs)
# ---------------------------------------------------------------------------

class TestResolveConnection:
    async def test_resolves_both_refs(self):
        n1 = _active_prop("src1", "source idea")
        n2 = _active_prop("tgt1", "target idea")
        graph = _graph_with(n1, n2)

        # search is called once per description; both return high scores
        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("src1", 0.91)],  # for source_node_description
            [("tgt1", 0.92)],  # for target_node_description
        ])

        event = ClassifiedEvent(
            event_type=EventType.CONNECTION,
            source_node_description="source idea",
            target_node_description="target idea",
            basis="they are related",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert drop_info is None
        assert result.payload["source_node_id"] == "src1"
        assert result.payload["target_node_id"] == "tgt1"
        assert result.payload["basis"] == "they are related"

    async def test_drops_if_source_unresolvable(self):
        n2 = _active_prop("tgt1", "target idea")
        graph = _graph_with(n2)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [],             # source not found
            [("tgt1", 0.9)],
        ])

        event = ClassifiedEvent(
            event_type=EventType.CONNECTION,
            source_node_description="missing source",
            target_node_description="target idea",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None

    async def test_drops_if_target_unresolvable(self):
        n1 = _active_prop("src1", "source idea")
        graph = _graph_with(n1)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("src1", 0.9)],
            [],             # target not found
        ])

        event = ClassifiedEvent(
            event_type=EventType.CONNECTION,
            source_node_description="source idea",
            target_node_description="missing target",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_classified_event: TENSION_IDENTIFIED (required list ref, min 2)
# ---------------------------------------------------------------------------

class TestResolveTensionIdentified:
    async def test_resolves_two_nodes(self):
        n1 = _active_prop("n1", "idea A")
        n2 = _active_prop("n2", "idea B")
        graph = _graph_with(n1, n2)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("n1", 0.90)],
            [("n2", 0.91)],
        ])

        event = ClassifiedEvent(
            event_type=EventType.TENSION_IDENTIFIED,
            node_descriptions=["idea A", "idea B"],
            description="These two ideas are in tension",
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert set(result.payload["node_ids"]) == {"n1", "n2"}
        assert result.payload["description"] == "These two ideas are in tension"

    async def test_drops_if_fewer_than_two_resolve(self):
        n1 = _active_prop("n1", "idea A")
        graph = _graph_with(n1)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("n1", 0.90)],
            [],  # second one doesn't resolve
        ])

        event = ClassifiedEvent(
            event_type=EventType.TENSION_IDENTIFIED,
            node_descriptions=["idea A", "missing idea"],
            description="tension",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_classified_event: SYNTHESIS (required list ref, min 2)
# ---------------------------------------------------------------------------

class TestResolveSynthesis:
    async def test_resolves_constituents(self):
        n1 = _active_prop("n1", "idea A")
        n2 = _active_prop("n2", "idea B")
        graph = _graph_with(n1, n2)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("n1", 0.93)],
            [("n2", 0.90)],
        ])

        event = ClassifiedEvent(
            event_type=EventType.SYNTHESIS,
            text="unified synthesis",
            constituent_node_descriptions=["idea A", "idea B"],
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert set(result.payload["constituent_node_ids"]) == {"n1", "n2"}
        assert result.payload["text"] == "unified synthesis"

    async def test_drops_if_fewer_than_two_resolve(self):
        n1 = _active_prop("n1", "idea A")
        graph = _graph_with(n1)

        idx = MagicMock()
        idx.search = AsyncMock(side_effect=[
            [("n1", 0.90)],
            [],
        ])

        event = ClassifiedEvent(
            event_type=EventType.SYNTHESIS,
            text="synthesis",
            constituent_node_descriptions=["idea A", "missing"],
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_classified_event: QUESTION_RESOLVED (required single ref)
# ---------------------------------------------------------------------------

class TestResolveQuestionResolved:
    async def test_resolves_question_ref(self):
        q_node = GraphNode(
            id="q1", type=NodeType.QUESTION, text="Will AI replace jobs?",
            status=NodeStatus.ACTIVE,
        )
        graph = _graph_with(q_node)
        idx = _mock_index([("q1", 0.91)])

        event = ClassifiedEvent(
            event_type=EventType.QUESTION_RESOLVED,
            question_node_description="Will AI replace jobs?",
            resolution_text="Partially, in some sectors",
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["question_node_id"] == "q1"
        assert result.payload["resolution_text"] == "Partially, in some sectors"

    async def test_drops_if_question_not_resolved(self):
        graph = IdeaGraph()
        idx = _mock_index([])
        event = ClassifiedEvent(
            event_type=EventType.QUESTION_RESOLVED,
            question_node_description="missing question",
            resolution_text="answer",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_classified_event: REFRAME (optional list ref)
# ---------------------------------------------------------------------------

class TestResolveReframe:
    async def test_resolves_affected_nodes(self):
        n1 = _active_prop("n1", "old framing")
        graph = _graph_with(n1)
        idx = _mock_index([("n1", 0.90)])

        event = ClassifiedEvent(
            event_type=EventType.REFRAME,
            text="new framing text",
            affected_node_descriptions=["old framing"],
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["affected_node_ids"] == ["n1"]

    async def test_resolves_with_no_affected_nodes(self):
        graph = IdeaGraph()
        idx = _mock_index([])

        event = ClassifiedEvent(
            event_type=EventType.REFRAME,
            text="new frame",
            source="user",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["affected_node_ids"] == []


# ---------------------------------------------------------------------------
# resolve_classified_event: ABANDONMENT (required single ref)
# ---------------------------------------------------------------------------

class TestResolveAbandonment:
    async def test_resolves_target(self):
        n1 = _active_prop("n1", "abandoned idea")
        graph = _graph_with(n1)
        idx = _mock_index([("n1", 0.88)])

        event = ClassifiedEvent(
            event_type=EventType.ABANDONMENT,
            target_node_description="abandoned idea",
            reason="no longer relevant",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.payload["target_node_id"] == "n1"
        assert result.payload["reason"] == "no longer relevant"

    async def test_drops_if_unresolvable(self):
        graph = IdeaGraph()
        idx = _mock_index([])
        event = ClassifiedEvent(
            event_type=EventType.ABANDONMENT,
            target_node_description="missing",
            reason="reason",
        )
        result, drop_info = await resolve_classified_event(event, graph, idx)
        assert result is None


# ---------------------------------------------------------------------------
# ClassifiedResult properties
# ---------------------------------------------------------------------------

class TestClassifiedResult:
    async def test_result_carries_reasoning(self):
        event = ClassifiedEvent(
            event_type=EventType.NEW_PROPOSITION,
            text="idea",
            reasoning="this is why",
        )
        graph = IdeaGraph()
        idx = _mock_index([])

        result, drop_info = await resolve_classified_event(event, graph, idx)

        assert result is not None
        assert result.reasoning == "this is why"
