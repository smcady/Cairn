"""Tests for IdeaGraph operations."""

import pytest

from cairn.models.graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    IdeaGraph,
    NodeStatus,
    NodeType,
)


@pytest.fixture
def graph():
    return IdeaGraph()


@pytest.fixture
def sample_node():
    return GraphNode(
        id="node1",
        type=NodeType.PROPOSITION,
        text="Test proposition",
        confidence=0.5,
        source="test",
    )


class TestIdeaGraph:
    def test_add_and_get_node(self, graph, sample_node):
        graph.add_node(sample_node)
        retrieved = graph.get_node("node1")
        assert retrieved is not None
        assert retrieved.id == "node1"
        assert retrieved.text == "Test proposition"
        assert retrieved.type == NodeType.PROPOSITION

    def test_get_nonexistent_node(self, graph):
        assert graph.get_node("nonexistent") is None

    def test_update_node(self, graph, sample_node):
        graph.add_node(sample_node)
        updated = graph.update_node("node1", text="Updated text", confidence=0.8)
        assert updated is not None
        assert updated.text == "Updated text"
        assert updated.confidence == 0.8

        # Verify persistence
        retrieved = graph.get_node("node1")
        assert retrieved.text == "Updated text"

    def test_update_nonexistent_node(self, graph):
        assert graph.update_node("nonexistent", text="x") is None

    def test_add_and_get_edge(self, graph):
        n1 = GraphNode(id="n1", type=NodeType.PROPOSITION, text="A")
        n2 = GraphNode(id="n2", type=NodeType.PROPOSITION, text="B")
        graph.add_node(n1)
        graph.add_node(n2)

        edge = GraphEdge(type=EdgeType.SUPPORTS, strength=0.7)
        key = graph.add_edge("n1", "n2", edge)
        assert key is not None

        edges = graph.get_edges("n1", "n2")
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS
        assert edges[0].strength == 0.7

    def test_add_edge_nonexistent_node(self, graph, sample_node):
        graph.add_node(sample_node)
        edge = GraphEdge(type=EdgeType.SUPPORTS)
        assert graph.add_edge("node1", "nonexistent", edge) is None

    def test_multi_edges(self, graph):
        n1 = GraphNode(id="n1", type=NodeType.PROPOSITION, text="A")
        n2 = GraphNode(id="n2", type=NodeType.PROPOSITION, text="B")
        graph.add_node(n1)
        graph.add_node(n2)

        graph.add_edge("n1", "n2", GraphEdge(type=EdgeType.SUPPORTS))
        graph.add_edge("n1", "n2", GraphEdge(type=EdgeType.RELATES_TO))

        edges = graph.get_edges("n1", "n2")
        assert len(edges) == 2

    def test_get_all_nodes(self, graph):
        graph.add_node(GraphNode(id="p1", type=NodeType.PROPOSITION, text="A"))
        graph.add_node(GraphNode(id="q1", type=NodeType.QUESTION, text="B?"))
        graph.add_node(GraphNode(id="p2", type=NodeType.PROPOSITION, text="C"))

        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == 3

        props = graph.get_all_nodes(NodeType.PROPOSITION)
        assert len(props) == 2

        questions = graph.get_all_nodes(NodeType.QUESTION)
        assert len(questions) == 1

    def test_get_node_neighbors(self, graph):
        for i in range(4):
            graph.add_node(GraphNode(id=f"n{i}", type=NodeType.PROPOSITION, text=f"N{i}"))

        graph.add_edge("n0", "n1", GraphEdge(type=EdgeType.SUPPORTS))
        graph.add_edge("n2", "n0", GraphEdge(type=EdgeType.CONTRADICTS))
        graph.add_edge("n0", "n3", GraphEdge(type=EdgeType.RELATES_TO))

        out_neighbors = graph.get_node_neighbors("n0", "out")
        assert set(out_neighbors) == {"n1", "n3"}

        in_neighbors = graph.get_node_neighbors("n0", "in")
        assert set(in_neighbors) == {"n2"}

        both = graph.get_node_neighbors("n0", "both")
        assert set(both) == {"n1", "n2", "n3"}

    def test_get_edges_for_node(self, graph):
        graph.add_node(GraphNode(id="n0", type=NodeType.PROPOSITION, text="Center"))
        graph.add_node(GraphNode(id="n1", type=NodeType.EVIDENCE, text="Evidence"))
        graph.add_node(GraphNode(id="n2", type=NodeType.OBJECTION, text="Objection"))

        graph.add_edge("n1", "n0", GraphEdge(type=EdgeType.SUPPORTS))
        graph.add_edge("n2", "n0", GraphEdge(type=EdgeType.CONTRADICTS))

        in_edges = graph.get_edges_for_node("n0", "in")
        assert len(in_edges) == 2

        edge_types = {e.type for _, _, e in in_edges}
        assert EdgeType.SUPPORTS in edge_types
        assert EdgeType.CONTRADICTS in edge_types

    def test_get_nodes_by_status(self, graph):
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="Active", status=NodeStatus.ACTIVE))
        graph.add_node(GraphNode(id="n2", type=NodeType.PROPOSITION, text="Parked", status=NodeStatus.PARKED))
        graph.add_node(GraphNode(id="n3", type=NodeType.QUESTION, text="Resolved", status=NodeStatus.RESOLVED))

        active = graph.get_nodes_by_status(NodeStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].id == "n1"

    def test_subgraph_around(self, graph):
        for i in range(5):
            graph.add_node(GraphNode(id=f"n{i}", type=NodeType.PROPOSITION, text=f"N{i}"))

        graph.add_edge("n0", "n1", GraphEdge(type=EdgeType.RELATES_TO))
        graph.add_edge("n1", "n2", GraphEdge(type=EdgeType.RELATES_TO))
        graph.add_edge("n2", "n3", GraphEdge(type=EdgeType.RELATES_TO))
        graph.add_edge("n3", "n4", GraphEdge(type=EdgeType.RELATES_TO))

        depth1 = graph.get_subgraph_around("n0", depth=1)
        assert "n0" in depth1
        assert "n1" in depth1
        assert "n2" not in depth1

        depth2 = graph.get_subgraph_around("n0", depth=2)
        assert "n2" in depth2
        assert "n3" not in depth2

    def test_node_count_edge_count(self, graph):
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="A"))
        graph.add_node(GraphNode(id="n2", type=NodeType.PROPOSITION, text="B"))
        assert graph.node_count() == 2

        graph.add_edge("n1", "n2", GraphEdge(type=EdgeType.RELATES_TO))
        assert graph.edge_count() == 1

    def test_clear(self, graph):
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="A"))
        graph.add_node(GraphNode(id="n2", type=NodeType.PROPOSITION, text="B"))
        graph.add_edge("n1", "n2", GraphEdge(type=EdgeType.RELATES_TO))

        graph.clear()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_node_summary_list(self, graph):
        graph.add_node(GraphNode(id="n1", type=NodeType.PROPOSITION, text="Test idea"))
        summaries = graph.node_summary_list()
        assert len(summaries) == 1
        assert summaries[0]["id"] == "n1"
        assert summaries[0]["type"] == "proposition"
        assert summaries[0]["text"] == "Test idea"
