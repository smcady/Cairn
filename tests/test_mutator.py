"""Tests for deterministic mutation rules."""

import pytest

from cairn.models.events import Event, EventType
from cairn.models.graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    IdeaGraph,
    NodeStatus,
    NodeType,
)
from cairn.pipeline.mutator import apply_event


@pytest.fixture
def graph():
    return IdeaGraph()


@pytest.fixture
def graph_with_proposition(graph):
    """Graph with a single proposition node at id='prop1'."""
    node = GraphNode(id="prop1", type=NodeType.PROPOSITION, text="Initial claim", confidence=0.5)
    graph.add_node(node)
    return graph


def _make_event(event_type: EventType, payload: dict, turn: int = 1) -> Event:
    return Event(
        event_type=event_type,
        payload=payload,
        turn_number=turn,
        session_id="test",
    )


class TestNewProposition:
    def test_creates_proposition_node(self, graph):
        event = _make_event(EventType.NEW_PROPOSITION, {
            "text": "AI will transform education",
            "source": "user",
            "related_node_ids": [],
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        node = graph.get_node(result.created_node_ids[0])
        assert node.type == NodeType.PROPOSITION
        assert node.text == "AI will transform education"
        assert node.confidence == 0.5

    def test_creates_relates_to_edges(self, graph_with_proposition):
        event = _make_event(EventType.NEW_PROPOSITION, {
            "text": "Education needs reform",
            "source": "user",
            "related_node_ids": ["prop1"],
        })
        result = apply_event(graph_with_proposition, event)

        assert len(result.created_edges) == 1
        assert result.created_edges[0][1] == "prop1"
        assert result.created_edges[0][2] == EdgeType.RELATES_TO

    def test_ignores_invalid_related_ids(self, graph):
        event = _make_event(EventType.NEW_PROPOSITION, {
            "text": "Some idea",
            "source": "user",
            "related_node_ids": ["nonexistent"],
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        assert len(result.created_edges) == 0


class TestSupport:
    def test_creates_evidence_and_increments_confidence(self, graph_with_proposition):
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "prop1",
            "evidence_text": "Studies show X",
            "source": "user",
        })
        result = apply_event(graph_with_proposition, event)

        assert len(result.created_node_ids) == 1
        evidence = graph_with_proposition.get_node(result.created_node_ids[0])
        assert evidence.type == NodeType.EVIDENCE
        assert evidence.text == "Studies show X"

        prop = graph_with_proposition.get_node("prop1")
        assert prop.confidence == 0.6  # 0.5 + 0.1

        assert len(result.created_edges) == 1
        assert result.created_edges[0][2] == EdgeType.SUPPORTS

    def test_confidence_caps_at_09(self, graph):
        node = GraphNode(id="high", type=NodeType.PROPOSITION, text="Strong claim", confidence=0.85)
        graph.add_node(node)

        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "high",
            "evidence_text": "More evidence",
            "source": "user",
        })
        apply_event(graph, event)

        assert graph.get_node("high").confidence == 0.9

    def test_no_op_for_missing_target(self, graph):
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "missing",
            "evidence_text": "Evidence",
            "source": "user",
        })
        result = apply_event(graph, event)
        assert len(result.created_node_ids) == 0


class TestContradiction:
    def test_creates_objection_and_decrements_confidence(self, graph_with_proposition):
        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "prop1",
            "objection_text": "But counterpoint Y",
            "source": "user",
        })
        result = apply_event(graph_with_proposition, event)

        assert len(result.created_node_ids) == 1
        objection = graph_with_proposition.get_node(result.created_node_ids[0])
        assert objection.type == NodeType.OBJECTION

        prop = graph_with_proposition.get_node("prop1")
        assert prop.confidence == 0.4  # 0.5 - 0.1

    def test_confidence_floors_at_01(self, graph):
        node = GraphNode(id="weak", type=NodeType.PROPOSITION, text="Weak claim", confidence=0.15)
        graph.add_node(node)

        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "weak",
            "objection_text": "Disagree",
            "source": "user",
        })
        apply_event(graph, event)

        assert graph.get_node("weak").confidence == 0.1

    def test_auto_tension_below_03(self, graph):
        node = GraphNode(id="contested", type=NodeType.PROPOSITION, text="Contested claim", confidence=0.25)
        graph.add_node(node)

        # Add an existing objection with CONTRADICTS edge
        objection = GraphNode(id="obj1", type=NodeType.OBJECTION, text="Prior objection")
        graph.add_node(objection)
        graph.add_edge("obj1", "contested", GraphEdge(type=EdgeType.CONTRADICTS, strength=0.5))

        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "contested",
            "objection_text": "Another objection",
            "source": "user",
        })
        result = apply_event(graph, event)

        # Should have created the new objection AND a tension node
        tensions = [nid for nid in result.created_node_ids
                    if graph.get_node(nid).type == NodeType.TENSION]
        assert len(tensions) == 1

        tension_node = graph.get_node(tensions[0])
        assert tension_node.status == NodeStatus.ACTIVE


class TestRefinement:
    def test_updates_text_and_preserves_history(self, graph_with_proposition):
        event = _make_event(EventType.REFINEMENT, {
            "target_node_id": "prop1",
            "new_text": "Refined claim with more precision",
            "source": "user",
        })
        result = apply_event(graph_with_proposition, event)

        assert "prop1" in result.modified_node_ids
        node = graph_with_proposition.get_node("prop1")
        assert node.text == "Refined claim with more precision"
        assert node.version_history == ["Initial claim"]
        assert node.depth_of_exploration == 1

    def test_multiple_refinements_build_history(self, graph_with_proposition):
        for i in range(3):
            event = _make_event(EventType.REFINEMENT, {
                "target_node_id": "prop1",
                "new_text": f"Version {i + 1}",
                "source": "user",
            })
            apply_event(graph_with_proposition, event)

        node = graph_with_proposition.get_node("prop1")
        assert node.text == "Version 3"
        assert len(node.version_history) == 3
        assert node.depth_of_exploration == 3


class TestNewQuestion:
    def test_creates_question_node(self, graph):
        event = _make_event(EventType.NEW_QUESTION, {
            "text": "What about edge cases?",
            "related_node_ids": [],
            "source": "user",
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        node = graph.get_node(result.created_node_ids[0])
        assert node.type == NodeType.QUESTION
        assert node.status == NodeStatus.ACTIVE

    def test_creates_questions_edges(self, graph_with_proposition):
        event = _make_event(EventType.NEW_QUESTION, {
            "text": "Does this apply to K-12?",
            "related_node_ids": ["prop1"],
            "source": "user",
        })
        result = apply_event(graph_with_proposition, event)

        assert len(result.created_edges) == 1
        assert result.created_edges[0][2] == EdgeType.QUESTIONS


class TestQuestionResolved:
    def test_resolves_question(self, graph):
        q = GraphNode(id="q1", type=NodeType.QUESTION, text="Open question?", status=NodeStatus.ACTIVE)
        graph.add_node(q)

        event = _make_event(EventType.QUESTION_RESOLVED, {
            "question_node_id": "q1",
            "resolution_text": "The answer is yes because...",
            "source": "user",
        })
        result = apply_event(graph, event)

        assert graph.get_node("q1").status == NodeStatus.RESOLVED
        assert len(result.created_node_ids) == 1  # resolution proposition

        resolution = graph.get_node(result.created_node_ids[0])
        assert resolution.type == NodeType.PROPOSITION
        assert len(result.created_edges) == 1
        assert result.created_edges[0][2] == EdgeType.RESOLVES


class TestConnection:
    def test_creates_relates_to_edge(self, graph):
        graph.add_node(GraphNode(id="a", type=NodeType.PROPOSITION, text="A"))
        graph.add_node(GraphNode(id="b", type=NodeType.PROPOSITION, text="B"))

        event = _make_event(EventType.CONNECTION, {
            "source_node_id": "a",
            "target_node_id": "b",
            "basis": "Both relate to education",
        })
        result = apply_event(graph, event)

        assert len(result.created_edges) == 1
        edges = graph.get_edges("a", "b")
        assert edges[0].basis == "Both relate to education"


class TestTensionIdentified:
    def test_creates_tension_with_between_edges(self, graph):
        graph.add_node(GraphNode(id="a", type=NodeType.PROPOSITION, text="Claim A"))
        graph.add_node(GraphNode(id="b", type=NodeType.PROPOSITION, text="Claim B"))

        event = _make_event(EventType.TENSION_IDENTIFIED, {
            "node_ids": ["a", "b"],
            "description": "A and B cannot both be true",
            "source": "user",
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        tension = graph.get_node(result.created_node_ids[0])
        assert tension.type == NodeType.TENSION
        assert len(result.created_edges) == 2  # BETWEEN edges to a and b


class TestTerritoryIdentified:
    def test_creates_territory_with_adjacent_edges(self, graph_with_proposition):
        event = _make_event(EventType.TERRITORY_IDENTIFIED, {
            "text": "Impact on teacher training",
            "adjacent_node_ids": ["prop1"],
            "source": "user",
        })
        result = apply_event(graph_with_proposition, event)

        assert len(result.created_node_ids) == 1
        territory = graph_with_proposition.get_node(result.created_node_ids[0])
        assert territory.type == NodeType.TERRITORY
        assert len(result.created_edges) == 1
        assert result.created_edges[0][2] == EdgeType.ADJACENT_TO


class TestReframe:
    def test_creates_frame_and_supersedes_prior(self, graph_with_proposition):
        # First frame
        frame1_event = _make_event(EventType.REFRAME, {
            "text": "Economic lens",
            "affected_node_ids": ["prop1"],
            "source": "user",
            "node_id": "frame1",
        })
        result1 = apply_event(graph_with_proposition, frame1_event)
        frame1_id = result1.created_node_ids[0]

        # Second frame on same node
        frame2_event = _make_event(EventType.REFRAME, {
            "text": "Social equity lens",
            "affected_node_ids": ["prop1"],
            "source": "user",
            "node_id": "frame2",
        })
        result2 = apply_event(graph_with_proposition, frame2_event)

        # First frame should be superseded
        assert graph_with_proposition.get_node(frame1_id).status == NodeStatus.SUPERSEDED
        assert frame1_id in result2.modified_node_ids


class TestSynthesis:
    def test_creates_synthesis_and_supersedes_constituents(self, graph):
        graph.add_node(GraphNode(id="a", type=NodeType.PROPOSITION, text="Thread A"))
        graph.add_node(GraphNode(id="b", type=NodeType.PROPOSITION, text="Thread B"))

        event = _make_event(EventType.SYNTHESIS, {
            "text": "A and B together mean C",
            "constituent_node_ids": ["a", "b"],
            "source": "user",
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        synthesis = graph.get_node(result.created_node_ids[0])
        assert synthesis.type == NodeType.SYNTHESIS
        assert len(result.created_edges) == 2

        # Default: constituents are superseded
        assert graph.get_node("a").status == NodeStatus.SUPERSEDED
        assert graph.get_node("b").status == NodeStatus.SUPERSEDED
        assert "a" in result.modified_node_ids
        assert "b" in result.modified_node_ids

    def test_additive_synthesis_preserves_constituents(self, graph):
        graph.add_node(GraphNode(id="a", type=NodeType.PROPOSITION, text="Thread A"))
        graph.add_node(GraphNode(id="b", type=NodeType.PROPOSITION, text="Thread B"))

        event = _make_event(EventType.SYNTHESIS, {
            "text": "A and B together also imply C",
            "constituent_node_ids": ["a", "b"],
            "source": "user",
            "supersedes_constituents": False,
        })
        result = apply_event(graph, event)

        assert len(result.created_node_ids) == 1
        assert len(result.created_edges) == 2
        assert graph.get_node("a").status == NodeStatus.ACTIVE
        assert graph.get_node("b").status == NodeStatus.ACTIVE
        assert "a" not in result.modified_node_ids

    def test_synthesis_skips_already_superseded(self, graph):
        graph.add_node(GraphNode(id="a", type=NodeType.PROPOSITION, text="Thread A", status=NodeStatus.SUPERSEDED))
        graph.add_node(GraphNode(id="b", type=NodeType.PROPOSITION, text="Thread B"))

        event = _make_event(EventType.SYNTHESIS, {
            "text": "A and B together mean C",
            "constituent_node_ids": ["a", "b"],
            "source": "user",
        })
        result = apply_event(graph, event)

        assert "a" not in result.modified_node_ids
        assert "b" in result.modified_node_ids


class TestAbandonment:
    def test_parks_target_node(self, graph_with_proposition):
        event = _make_event(EventType.ABANDONMENT, {
            "target_node_id": "prop1",
            "reason": "No longer relevant",
        })
        result = apply_event(graph_with_proposition, event)

        assert "prop1" in result.modified_node_ids
        assert graph_with_proposition.get_node("prop1").status == NodeStatus.PARKED


class TestEvidenceStrength:
    """Tests for variable confidence deltas based on evidence_strength."""

    def test_strong_support_large_delta(self, graph_with_proposition):
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "prop1",
            "evidence_text": "Benchmark shows 3x improvement",
            "source": "user",
            "evidence_strength": 0.8,
        })
        apply_event(graph_with_proposition, event)

        prop = graph_with_proposition.get_node("prop1")
        # 0.5 + (0.8 * 0.2) = 0.5 + 0.16 = 0.66
        assert abs(prop.confidence - 0.66) < 0.001

    def test_weak_support_small_delta(self, graph_with_proposition):
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "prop1",
            "evidence_text": "I think this is probably true",
            "source": "user",
            "evidence_strength": 0.1,
        })
        apply_event(graph_with_proposition, event)

        prop = graph_with_proposition.get_node("prop1")
        # 0.5 + (0.1 * 0.2) = 0.5 + 0.02 = 0.52
        assert abs(prop.confidence - 0.52) < 0.001

    def test_strong_contradiction_large_delta(self, graph_with_proposition):
        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "prop1",
            "objection_text": "Data shows 91% vs 62% retention gap",
            "source": "user",
            "evidence_strength": 0.8,
        })
        apply_event(graph_with_proposition, event)

        prop = graph_with_proposition.get_node("prop1")
        # 0.5 - (0.8 * 0.2) = 0.5 - 0.16 = 0.34
        assert abs(prop.confidence - 0.34) < 0.001

    def test_weak_contradiction_small_delta(self, graph_with_proposition):
        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "prop1",
            "objection_text": "I disagree",
            "source": "user",
            "evidence_strength": 0.2,
        })
        apply_event(graph_with_proposition, event)

        prop = graph_with_proposition.get_node("prop1")
        # 0.5 - (0.2 * 0.2) = 0.5 - 0.04 = 0.46
        assert abs(prop.confidence - 0.46) < 0.001

    def test_default_strength_preserves_old_behavior(self, graph_with_proposition):
        """Default evidence_strength=0.5 produces delta of 0.1, same as old fixed behavior."""
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "prop1",
            "evidence_text": "Some evidence",
            "source": "user",
            # no evidence_strength = defaults to 0.5
        })
        apply_event(graph_with_proposition, event)

        prop = graph_with_proposition.get_node("prop1")
        # 0.5 + (0.5 * 0.2) = 0.5 + 0.1 = 0.6
        assert prop.confidence == 0.6

    def test_edge_strength_set_from_evidence_strength(self, graph_with_proposition):
        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "prop1",
            "evidence_text": "Strong data",
            "source": "user",
            "evidence_strength": 0.75,
        })
        result = apply_event(graph_with_proposition, event)

        evidence_id = result.created_node_ids[0]
        edges = graph_with_proposition.get_edges(evidence_id, "prop1")
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS
        assert edges[0].strength == 0.75

    def test_contradiction_edge_strength_set(self, graph_with_proposition):
        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "prop1",
            "objection_text": "Counter evidence",
            "source": "user",
            "evidence_strength": 0.6,
        })
        result = apply_event(graph_with_proposition, event)

        objection_id = result.created_node_ids[0]
        edges = graph_with_proposition.get_edges(objection_id, "prop1")
        assert len(edges) == 1
        assert edges[0].type == EdgeType.CONTRADICTS
        assert edges[0].strength == 0.6

    def test_max_strength_support_caps_at_09(self, graph):
        node = GraphNode(id="high", type=NodeType.PROPOSITION, text="Strong claim", confidence=0.8)
        graph.add_node(node)

        event = _make_event(EventType.SUPPORT, {
            "target_node_id": "high",
            "evidence_text": "Proven result",
            "source": "user",
            "evidence_strength": 1.0,
        })
        apply_event(graph, event)

        # 0.8 + (1.0 * 0.2) = 1.0 -> capped at 0.9
        assert graph.get_node("high").confidence == 0.9

    def test_max_strength_contradiction_floors_at_01(self, graph):
        node = GraphNode(id="weak", type=NodeType.PROPOSITION, text="Weak claim", confidence=0.2)
        graph.add_node(node)

        event = _make_event(EventType.CONTRADICTION, {
            "target_node_id": "weak",
            "objection_text": "Definitive disproof",
            "source": "user",
            "evidence_strength": 1.0,
        })
        apply_event(graph, event)

        # 0.2 - (1.0 * 0.2) = 0.0 -> floored at 0.1
        assert graph.get_node("weak").confidence == 0.1
