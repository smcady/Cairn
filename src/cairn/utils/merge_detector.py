"""Background redundant node detection and merging via text and semantic similarity."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from cairn.models.graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    IdeaGraph,
    NodeStatus,
)

if TYPE_CHECKING:
    from cairn.utils.vector_index import VectorIndex


def text_similarity(a: str, b: str) -> float:
    """Compute text similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_merge_candidates(
    graph: IdeaGraph,
    threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    """Find pairs of nodes that are likely duplicates based on text similarity.

    Only considers active nodes of the same type.

    Args:
        graph: The idea graph
        threshold: Minimum similarity score (0-1) to consider a merge

    Returns:
        List of (node_id_1, node_id_2, similarity_score) tuples
    """
    active_nodes = [n for n in graph.get_all_nodes() if n.status == NodeStatus.ACTIVE]
    candidates = []

    for i, node_a in enumerate(active_nodes):
        for node_b in active_nodes[i + 1:]:
            # Only compare nodes of the same type
            if node_a.type != node_b.type:
                continue

            sim = text_similarity(node_a.text, node_b.text)
            if sim >= threshold:
                candidates.append((node_a.id, node_b.id, sim))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def merge_nodes(graph: IdeaGraph, keep_id: str, remove_id: str) -> bool:
    """Merge two nodes, keeping one and transferring all edges from the other.

    The kept node absorbs all edges from the removed node. The removed node
    is marked as superseded.

    Args:
        graph: The idea graph
        keep_id: ID of the node to keep
        remove_id: ID of the node to remove (will be superseded)

    Returns:
        True if merge was successful
    """
    keep_node = graph.get_node(keep_id)
    remove_node = graph.get_node(remove_id)

    if not keep_node or not remove_node:
        return False

    # Transfer all edges from the removed node to the kept node
    # Outgoing edges
    for _u, v, edge in graph.get_edges_for_node(remove_id, direction="out"):
        if v != keep_id:  # Don't create self-loops
            graph.add_edge(keep_id, v, edge)

    # Incoming edges
    for u, _v, edge in graph.get_edges_for_node(remove_id, direction="in"):
        if u != keep_id:  # Don't create self-loops
            graph.add_edge(u, keep_id, edge)

    # Keep the higher confidence
    if remove_node.confidence > keep_node.confidence:
        graph.update_node(keep_id, confidence=remove_node.confidence)

    # Keep the higher exploration depth
    if remove_node.depth_of_exploration > keep_node.depth_of_exploration:
        graph.update_node(keep_id, depth_of_exploration=remove_node.depth_of_exploration)

    # Merge version histories
    merged_history = list(keep_node.version_history)
    if remove_node.text not in merged_history and remove_node.text != keep_node.text:
        merged_history.append(remove_node.text)
    graph.update_node(keep_id, version_history=merged_history)

    # Mark the removed node as superseded
    graph.update_node(remove_id, status=NodeStatus.SUPERSEDED)

    return True


def run_merge_pass(
    graph: IdeaGraph,
    threshold: float = 0.8,
) -> list[tuple[str, str]]:
    """Run a single merge detection pass.

    Finds candidates and merges them (keeping the node with more edges).

    Args:
        graph: The idea graph
        threshold: Minimum similarity for merge

    Returns:
        List of (kept_id, removed_id) pairs that were merged
    """
    candidates = find_merge_candidates(graph, threshold)
    merged = []
    already_merged: set[str] = set()

    for id_a, id_b, _sim in candidates:
        if id_a in already_merged or id_b in already_merged:
            continue

        # Keep the node with more connections
        edges_a = len(graph.get_edges_for_node(id_a))
        edges_b = len(graph.get_edges_for_node(id_b))

        if edges_a >= edges_b:
            keep_id, remove_id = id_a, id_b
        else:
            keep_id, remove_id = id_b, id_a

        if merge_nodes(graph, keep_id, remove_id):
            merged.append((keep_id, remove_id))
            already_merged.add(remove_id)

    return merged


def find_merge_candidates_semantic(
    graph: IdeaGraph,
    vector_index: "VectorIndex",
    threshold: float = 0.92,
) -> list[tuple[str, str, float]]:
    """Find merge candidates using semantic (vector) similarity.

    Uses cached embeddings — no API calls. Higher threshold than lexical (0.92 vs 0.80)
    to avoid false positives on semantically related but distinct ideas.
    Only considers active nodes of the same type that are both indexed.
    """
    active_nodes = [n for n in graph.get_all_nodes() if n.status == NodeStatus.ACTIVE]
    indexed = vector_index.indexed_ids()
    candidates = []

    for i, node_a in enumerate(active_nodes):
        if node_a.id not in indexed:
            continue
        for node_b in active_nodes[i + 1:]:
            if node_b.id not in indexed:
                continue
            if node_a.type != node_b.type:
                continue
            sim = vector_index.cosine_similarity(node_a.id, node_b.id)
            if sim is not None and sim >= threshold:
                candidates.append((node_a.id, node_b.id, sim))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def run_merge_pass_with_index(
    graph: IdeaGraph,
    vector_index: "VectorIndex",
    lex_threshold: float = 0.8,
    sem_threshold: float = 0.92,
) -> list[tuple[str, str]]:
    """Run lexical + semantic merge detection and merge candidates.

    Runs both passes, deduplicates across them, merges. Existing run_merge_pass
    (lexical-only) is unchanged for backward compatibility.
    """
    lex_candidates = find_merge_candidates(graph, lex_threshold)
    sem_candidates = find_merge_candidates_semantic(graph, vector_index, sem_threshold)

    seen_pairs: set[frozenset] = set()
    all_candidates: list[tuple[str, str, float]] = []
    for id_a, id_b, score in lex_candidates + sem_candidates:
        pair = frozenset([id_a, id_b])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            all_candidates.append((id_a, id_b, score))

    all_candidates.sort(key=lambda x: x[2], reverse=True)

    merged = []
    already_merged: set[str] = set()
    for id_a, id_b, _score in all_candidates:
        if id_a in already_merged or id_b in already_merged:
            continue
        edges_a = len(graph.get_edges_for_node(id_a))
        edges_b = len(graph.get_edges_for_node(id_b))
        keep_id, remove_id = (id_a, id_b) if edges_a >= edges_b else (id_b, id_a)
        if merge_nodes(graph, keep_id, remove_id):
            merged.append((keep_id, remove_id))
            already_merged.add(remove_id)

    return merged
