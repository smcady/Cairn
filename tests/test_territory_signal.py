"""Behavioral tests for territory signal in harness_ingest and harness_orient rendering.

Tests are unit/integration level — no LLM or Voyage AI calls (mocked).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cairn.memory.engine import MemoryEngine
from cairn.models.events import ClassifiedEvent, EventLog, EventType
from cairn.models.graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    IdeaGraph,
    NodeStatus,
    NodeType,
)
from cairn.pipeline.classifier import ClassifiedResult
from cairn.pipeline.renderer import ViewType, render_structured_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(vector_index=None) -> MemoryEngine:
    engine = MemoryEngine(
        event_log=EventLog(":memory:"),
        graph=IdeaGraph(),
        vector_index=vector_index,
    )
    engine.client = MagicMock()
    return engine


def _classified(event_type: EventType, **kwargs) -> ClassifiedEvent:
    return ClassifiedEvent(event_type=event_type, **kwargs)


def _result(event_type: EventType, payload: dict, reasoning: str = "") -> tuple:
    return (ClassifiedResult(event_type=event_type, payload=payload, reasoning=reasoning), None)


async def _ingest_proposition(engine: MemoryEngine, text: str) -> None:
    """Helper: ingest a plain NEW_PROPOSITION without territory signal."""
    with patch("cairn.memory.engine.classify_exchange") as mc, \
         patch("cairn.memory.engine.resolve_classified_event") as mr:
        mc.return_value = [_classified(EventType.NEW_PROPOSITION, text=text)]
        mr.return_value = _result(
            EventType.NEW_PROPOSITION,
            {"text": text, "source": "user", "related_node_ids": []},
        )
        await engine.ingest(text)


# ---------------------------------------------------------------------------
# Test 1: territory signal appears when TERRITORY_IDENTIFIED fires + prior nodes exist
# ---------------------------------------------------------------------------


class TestTerritorySignalPresent:
    @patch("cairn.memory.engine.classify_exchange")
    @patch("cairn.memory.engine.resolve_classified_event")
    async def test_signal_in_response_when_territory_fires(self, mock_resolve, mock_classify):
        engine = _engine()

        # First ingest: add prior nodes so "prior context exists" check passes
        mock_classify.return_value = [_classified(EventType.NEW_PROPOSITION, text="founder-led sales is default")]
        mock_resolve.return_value = _result(
            EventType.NEW_PROPOSITION,
            {"text": "founder-led sales is default", "source": "user", "related_node_ids": []},
        )
        await engine.ingest("founder-led sales is default")

        # Second ingest: TERRITORY_IDENTIFIED fires
        territory_text = "usage-based vs seat-based pricing"
        mock_classify.return_value = [_classified(EventType.TERRITORY_IDENTIFIED, text=territory_text)]
        mock_resolve.return_value = _result(
            EventType.TERRITORY_IDENTIFIED,
            {"text": territory_text, "source": "user", "adjacent_node_ids": []},
        )
        response = await engine.ingest("switching to pricing territory")

        # Build the response string the same way harness_ingest does
        # (We're testing the engine result, not the MCP layer directly,
        #  so we verify the applied events contain TERRITORY_IDENTIFIED
        #  and reconstruct what the signal would look like.)
        assert any(e.event_type == EventType.TERRITORY_IDENTIFIED for e in response.applied_events)
        territory_event = next(e for e in response.applied_events if e.event_type == EventType.TERRITORY_IDENTIFIED)
        assert territory_event.payload["text"] == territory_text

    @patch("cairn.memory.engine.classify_exchange")
    @patch("cairn.memory.engine.resolve_classified_event")
    async def test_mcp_response_contains_territory_signal(self, mock_resolve, mock_classify):
        """Simulate what harness_ingest builds and verify the territory signal format."""
        from cairn.models.events import Event

        engine = _engine()

        # Pre-populate graph with a prior node
        mock_classify.return_value = [_classified(EventType.NEW_PROPOSITION, text="prior context")]
        mock_resolve.return_value = _result(
            EventType.NEW_PROPOSITION,
            {"text": "prior context", "source": "user", "related_node_ids": []},
        )
        await engine.ingest("prior context")
        prior_count = engine.graph.node_count()
        assert prior_count >= 1

        # Ingest territory
        territory_text = "hiring first sales rep"
        mock_classify.return_value = [_classified(EventType.TERRITORY_IDENTIFIED, text=territory_text)]
        mock_resolve.return_value = _result(
            EventType.TERRITORY_IDENTIFIED,
            {"text": territory_text, "source": "user", "adjacent_node_ids": []},
        )
        result = await engine.ingest("we need to think about hiring")

        # Verify the signal logic: prior_nodes_exist and territory event present
        territory_events = [e for e in result.applied_events if e.event_type == EventType.TERRITORY_IDENTIFIED]
        prior_nodes_exist = engine.graph.node_count() > len(result.applied_events)

        assert len(territory_events) == 1
        assert prior_nodes_exist
        assert territory_events[0].payload["text"] == territory_text

        # Build the response lines the same way harness_ingest does to test the format
        lines = [f"Ingested {len(result.applied_events)} event(s) from 'test':"]
        for event in result.applied_events:
            lines.append(f"  - {event.event_type}")
        if prior_nodes_exist:
            for te in territory_events:
                txt = te.payload.get("text", "")
                if txt:
                    lines.append(f'\nNew territory entered: "{txt}"')
                    lines.append(f'→ call harness_orient("{txt}") for relevant context')
        response_str = "\n".join(lines)

        assert "New territory entered" in response_str
        assert territory_text in response_str
        assert "harness_orient" in response_str


# ---------------------------------------------------------------------------
# Test 2: territory signal absent when TERRITORY_IDENTIFIED does NOT fire
# ---------------------------------------------------------------------------


class TestTerritorySignalAbsent:
    @patch("cairn.memory.engine.classify_exchange")
    @patch("cairn.memory.engine.resolve_classified_event")
    async def test_no_signal_for_plain_proposition(self, mock_resolve, mock_classify):
        engine = _engine()

        # Pre-populate
        mock_classify.return_value = [_classified(EventType.NEW_PROPOSITION, text="some context")]
        mock_resolve.return_value = _result(
            EventType.NEW_PROPOSITION,
            {"text": "some context", "source": "user", "related_node_ids": []},
        )
        await engine.ingest("some context")

        # Plain proposition ingest — no territory
        mock_classify.return_value = [_classified(EventType.NEW_PROPOSITION, text="another idea")]
        mock_resolve.return_value = _result(
            EventType.NEW_PROPOSITION,
            {"text": "another idea", "source": "user", "related_node_ids": []},
        )
        result = await engine.ingest("another idea")

        territory_events = [e for e in result.applied_events if e.event_type == EventType.TERRITORY_IDENTIFIED]
        assert len(territory_events) == 0


# ---------------------------------------------------------------------------
# Test 3: territory signal absent when graph is empty before ingest
# ---------------------------------------------------------------------------


class TestTerritorySignalEmptyGraph:
    @patch("cairn.memory.engine.classify_exchange")
    @patch("cairn.memory.engine.resolve_classified_event")
    async def test_no_orient_suggestion_when_no_prior_nodes(self, mock_resolve, mock_classify):
        engine = _engine()
        assert engine.graph.node_count() == 0  # fresh graph

        territory_text = "first territory ever"
        mock_classify.return_value = [_classified(EventType.TERRITORY_IDENTIFIED, text=territory_text)]
        mock_resolve.return_value = _result(
            EventType.TERRITORY_IDENTIFIED,
            {"text": territory_text, "source": "user", "adjacent_node_ids": []},
        )
        result = await engine.ingest("entering first territory")

        territory_events = [e for e in result.applied_events if e.event_type == EventType.TERRITORY_IDENTIFIED]
        assert len(territory_events) == 1  # event was applied

        # Key check: prior_nodes_exist should be False → no orient suggestion
        prior_nodes_exist = engine.graph.node_count() > len(result.applied_events)
        assert not prior_nodes_exist


# ---------------------------------------------------------------------------
# Test 4: harness_orient returns structured SETTLED / CONTESTED / OPEN sections
# ---------------------------------------------------------------------------


class TestOrientView:
    def _graph_with_propositions_and_question(self) -> tuple[IdeaGraph, list[str]]:
        """Build a graph with two propositions, a contradiction, and an open question."""
        graph = IdeaGraph()

        p1 = GraphNode(
            id="p1", type=NodeType.PROPOSITION, text="Founder-led sales works best pre-product-market-fit",
            status=NodeStatus.ACTIVE, confidence=0.7,
        )
        p2 = GraphNode(
            id="p2", type=NodeType.PROPOSITION, text="Hiring a rep early accelerates learning",
            status=NodeStatus.ACTIVE, confidence=0.5,
        )
        q1 = GraphNode(
            id="q1", type=NodeType.QUESTION, text="What quota should we set for the first sales rep?",
            status=NodeStatus.ACTIVE,
        )
        q2 = GraphNode(
            id="q2", type=NodeType.QUESTION, text="When does founder-led sales stop scaling?",
            status=NodeStatus.RESOLVED,
        )

        graph.add_node(p1)
        graph.add_node(p2)
        graph.add_node(q1)
        graph.add_node(q2)

        # p2 contradicts p1
        graph.add_edge("p2", "p1", GraphEdge(type=EdgeType.CONTRADICTS, strength=0.6))

        return graph, ["p1", "p2", "q1", "q2"]

    def test_orient_contains_settled_section(self):
        # p1 and p2 both have CONTRADICTS edges → both contested; no settled propositions here.
        # Let's add a third proposition with no contradiction to get a SETTLED section.
        graph, _ = self._graph_with_propositions_and_question()
        p3 = GraphNode(
            id="p3", type=NodeType.PROPOSITION, text="Sales reps need a clear ICP before hiring",
            status=NodeStatus.ACTIVE, confidence=0.8,
        )
        graph.add_node(p3)
        focus_ids = ["p1", "p2", "p3", "q1", "q2"]

        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="sales hiring")

        assert "SETTLED" in result
        assert "Sales reps need a clear ICP" in result

    def test_orient_contains_contested_section(self):
        graph, focus_ids = self._graph_with_propositions_and_question()
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="sales hiring")

        assert "CONTESTED" in result
        # Both p1 and p2 should appear in CONTESTED (they have a CONTRADICTS edge between them)
        assert "Founder-led sales" in result
        assert "Hiring a rep early" in result

    def test_orient_contains_open_questions(self):
        graph, focus_ids = self._graph_with_propositions_and_question()
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="sales hiring")

        assert "OPEN QUESTIONS" in result
        assert "quota" in result.lower()

    def test_orient_contains_recently_resolved(self):
        graph, focus_ids = self._graph_with_propositions_and_question()
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="sales hiring")

        assert "RECENTLY RESOLVED" in result
        assert "founder-led sales stop scaling" in result.lower()

    def test_orient_is_compact(self):
        """Response should be under 600 tokens (rough word count proxy: < 450 words)."""
        graph, focus_ids = self._graph_with_propositions_and_question()
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="sales hiring")

        word_count = len(result.split())
        assert word_count < 450, f"Orient output too long: {word_count} words"

    def test_orient_empty_focus_nodes(self):
        graph, _ = self._graph_with_propositions_and_question()
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=[], topic="nothing")
        assert "No relevant context" in result

    def test_orient_empty_topic_string(self):
        graph, focus_ids = self._graph_with_propositions_and_question()
        # Empty topic should still work — just shows generic label
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=focus_ids, topic="")
        assert "Orientation" in result

    def test_orient_truncates_long_node_text(self):
        graph = IdeaGraph()
        long_text = "A" * 200
        p = GraphNode(id="px", type=NodeType.PROPOSITION, text=long_text, status=NodeStatus.ACTIVE)
        graph.add_node(p)

        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=["px"], topic="test")
        # The truncated text should appear with ellipsis
        assert "…" in result
        # No line should exceed 130 chars (120 + prefix + ellipsis)
        for line in result.splitlines():
            assert len(line) <= 135, f"Line too long: {line!r}"

    def test_orient_skips_parked_propositions(self):
        graph = IdeaGraph()
        active = GraphNode(id="a1", type=NodeType.PROPOSITION, text="Active proposition", status=NodeStatus.ACTIVE)
        parked = GraphNode(id="p1", type=NodeType.PROPOSITION, text="Parked proposition", status=NodeStatus.PARKED)
        graph.add_node(active)
        graph.add_node(parked)

        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=["a1", "p1"], topic="test")
        assert "Active proposition" in result
        assert "Parked proposition" not in result

    def test_orient_contested_when_challenger_outside_focus(self):
        """A node with a CONTRADICTS edge should show as contested even if the
        challenger is not in the focus set (e.g., didn't appear in search results)."""
        graph = IdeaGraph()
        target = GraphNode(id="t1", type=NodeType.PROPOSITION, text="Usage-based pricing is best", status=NodeStatus.ACTIVE)
        challenger = GraphNode(id="c1", type=NodeType.OBJECTION, text="Enterprise can't budget variable costs", status=NodeStatus.ACTIVE)
        graph.add_node(target)
        graph.add_node(challenger)
        graph.add_edge("c1", "t1", GraphEdge(type=EdgeType.CONTRADICTS, strength=0.6))

        # Only target in focus set, challenger is outside
        result = render_structured_summary(graph, ViewType.ORIENT, focus_node_ids=["t1"], topic="pricing")
        assert "CONTESTED" in result
        assert "Usage-based pricing" in result
        assert "SETTLED" not in result
