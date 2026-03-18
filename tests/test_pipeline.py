"""Integration tests for the pipeline and merge detector."""

import pytest

from cairn.models.graph_types import (
    GraphNode,
    IdeaGraph,
    NodeStatus,
    NodeType,
)
from cairn.utils.merge_detector import (
    find_merge_candidates,
    merge_nodes,
    run_merge_pass,
    text_similarity,
)


class TestTextSimilarity:
    def test_identical_strings(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert text_similarity("abc", "xyz") < 0.5

    def test_similar_strings(self):
        sim = text_similarity(
            "AI will transform education",
            "AI will transform the education system",
        )
        assert sim > 0.7

    def test_case_insensitive(self):
        assert text_similarity("Hello World", "hello world") == 1.0


class TestFindMergeCandidates:
    def test_finds_similar_nodes(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(
            id="n1", type=NodeType.PROPOSITION,
            text="AI will transform education",
        ))
        graph.add_node(GraphNode(
            id="n2", type=NodeType.PROPOSITION,
            text="AI will transform the education system",
        ))
        graph.add_node(GraphNode(
            id="n3", type=NodeType.PROPOSITION,
            text="Something completely different about economics",
        ))

        candidates = find_merge_candidates(graph, threshold=0.7)
        assert len(candidates) == 1
        assert set(candidates[0][:2]) == {"n1", "n2"}

    def test_ignores_different_types(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(
            id="n1", type=NodeType.PROPOSITION,
            text="AI will transform education",
        ))
        graph.add_node(GraphNode(
            id="n2", type=NodeType.QUESTION,
            text="AI will transform education?",
        ))

        candidates = find_merge_candidates(graph, threshold=0.7)
        assert len(candidates) == 0

    def test_ignores_non_active_nodes(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(
            id="n1", type=NodeType.PROPOSITION,
            text="Same text here",
        ))
        graph.add_node(GraphNode(
            id="n2", type=NodeType.PROPOSITION,
            text="Same text here",
            status=NodeStatus.PARKED,
        ))

        candidates = find_merge_candidates(graph, threshold=0.7)
        assert len(candidates) == 0


class TestMergeNodes:
    def test_basic_merge(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(id="keep", type=NodeType.PROPOSITION, text="Keep me", confidence=0.6))
        graph.add_node(GraphNode(id="remove", type=NodeType.PROPOSITION, text="Remove me", confidence=0.4))

        result = merge_nodes(graph, "keep", "remove")
        assert result is True

        assert graph.get_node("remove").status == NodeStatus.SUPERSEDED
        assert graph.get_node("keep").confidence == 0.6  # keeps higher
        assert "Remove me" in graph.get_node("keep").version_history

    def test_merge_transfers_higher_confidence(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(id="keep", type=NodeType.PROPOSITION, text="A", confidence=0.3))
        graph.add_node(GraphNode(id="remove", type=NodeType.PROPOSITION, text="B", confidence=0.8))

        merge_nodes(graph, "keep", "remove")
        assert graph.get_node("keep").confidence == 0.8

    def test_merge_nonexistent_fails(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="A"))

        assert merge_nodes(graph, "n1", "nonexistent") is False


class TestRunMergePass:
    def test_merges_duplicates(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="The sky is blue"))
        graph.add_node(GraphNode(id="n2", type=NodeType.PROPOSITION, text="The sky is blue today"))

        merged = run_merge_pass(graph, threshold=0.7)
        assert len(merged) == 1

    def test_no_merges_when_distinct(self):
        graph = IdeaGraph()
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="AI transforms education"))
        graph.add_node(GraphNode(id="n2", type=NodeType.PROPOSITION, text="Quantum computing is fast"))

        merged = run_merge_pass(graph, threshold=0.7)
        assert len(merged) == 0
