"""Pydantic models for node/edge types and IdeaGraph wrapper around NetworkX MultiDiGraph."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field


# --- Enums ---

class NodeType(str, Enum):
    PROPOSITION = "proposition"
    QUESTION = "question"
    TENSION = "tension"
    TERRITORY = "territory"
    EVIDENCE = "evidence"
    OBJECTION = "objection"
    SYNTHESIS = "synthesis"
    FRAME = "frame"
    ABSTRACTION = "abstraction"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"
    PARKED = "parked"


class EdgeType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUESTIONS = "QUESTIONS"
    RELATES_TO = "RELATES_TO"
    REFRAMES = "REFRAMES"
    SYNTHESIZES = "SYNTHESIZES"
    ABSTRACTS_FROM = "ABSTRACTS_FROM"
    RESOLVES = "RESOLVES"
    BETWEEN = "BETWEEN"
    ADJACENT_TO = "ADJACENT_TO"


# --- Node & Edge Models ---

class GraphNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: NodeType
    text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "system"
    context: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: NodeStatus = NodeStatus.ACTIVE
    depth_of_exploration: int = 0
    version_history: list[str] = Field(default_factory=list)
    workspace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls.model_validate(data)


class GraphEdge(BaseModel):
    type: EdgeType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    basis: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# --- IdeaGraph Wrapper ---

class IdeaGraph:
    """Wrapper around NetworkX MultiDiGraph providing typed operations."""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def add_node(self, node: GraphNode) -> str:
        self._graph.add_node(node.id, **node.to_dict())
        return node.id

    def get_node(self, node_id: str) -> GraphNode | None:
        if node_id not in self._graph:
            return None
        return GraphNode.from_dict(self._graph.nodes[node_id])

    def update_node(self, node_id: str, **updates: Any) -> GraphNode | None:
        if node_id not in self._graph:
            return None
        data = dict(self._graph.nodes[node_id])
        data.update(updates)
        self._graph.nodes[node_id].update(updates)
        return GraphNode.from_dict(data)

    def add_edge(self, source_id: str, target_id: str, edge: GraphEdge) -> int | None:
        if source_id not in self._graph or target_id not in self._graph:
            return None
        key = self._graph.add_edge(source_id, target_id, **edge.to_dict())
        return key

    def get_edges(self, source_id: str, target_id: str) -> list[GraphEdge]:
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return []
        edges = []
        for _key, data in self._graph[source_id][target_id].items():
            edges.append(GraphEdge.model_validate(data))
        return edges

    def get_all_nodes(self, node_type: NodeType | None = None) -> list[GraphNode]:
        nodes = []
        for _nid, data in self._graph.nodes(data=True):
            node = GraphNode.from_dict(data)
            if node_type is None or node.type == node_type:
                nodes.append(node)
        return nodes

    def get_node_neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        if node_id not in self._graph:
            return []
        neighbors = set()
        if direction in ("out", "both"):
            neighbors.update(self._graph.successors(node_id))
        if direction in ("in", "both"):
            neighbors.update(self._graph.predecessors(node_id))
        return list(neighbors)

    def get_edges_for_node(self, node_id: str, direction: str = "both") -> list[tuple[str, str, GraphEdge]]:
        if node_id not in self._graph:
            return []
        edges = []
        if direction in ("out", "both"):
            for _u, v, data in self._graph.out_edges(node_id, data=True):
                edges.append((node_id, v, GraphEdge.model_validate(data)))
        if direction in ("in", "both"):
            for u, _v, data in self._graph.in_edges(node_id, data=True):
                edges.append((u, node_id, GraphEdge.model_validate(data)))
        return edges

    def get_nodes_by_status(self, status: NodeStatus) -> list[GraphNode]:
        return [n for n in self.get_all_nodes() if n.status == status]

    def get_subgraph_around(self, node_id: str, depth: int = 2) -> list[str]:
        """BFS from node_id up to `depth` hops, return list of node IDs."""
        if node_id not in self._graph:
            return []
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                for neighbor in self.get_node_neighbors(nid):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return list(visited)

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def clear(self) -> None:
        self._graph.clear()

    def get_nodes_by_workspace(self, workspace_id: str) -> list[GraphNode]:
        """Return all nodes belonging to a specific workspace."""
        return [n for n in self.get_all_nodes() if n.workspace_id == workspace_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph (nodes + edges) to a JSON-compatible dict."""
        nodes = {}
        for nid, data in self._graph.nodes(data=True):
            nodes[nid] = dict(data)
        edges = []
        for u, v, key, data in self._graph.edges(data=True, keys=True):
            edges.append({"source": u, "target": v, "key": key, **data})
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdeaGraph:
        """Deserialize a graph from a dict produced by to_dict()."""
        g = cls()
        for nid, node_data in data.get("nodes", {}).items():
            g._graph.add_node(nid, **node_data)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            edge.pop("key", None)
            g._graph.add_edge(src, tgt, **edge)
        return g

    def node_summary_list(self, workspace_id: str | None = None) -> list[dict[str, str]]:
        """Return a lightweight list of node id/type/text for the classifier context.

        When workspace_id is provided, only nodes from that workspace are included.
        """
        nodes = self.get_nodes_by_workspace(workspace_id) if workspace_id is not None else self.get_all_nodes()
        return [
            {"id": n.id, "type": n.type.value, "text": n.text, "status": n.status.value}
            for n in nodes
        ]
