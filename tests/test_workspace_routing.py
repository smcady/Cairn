"""Integration tests for workspace routing through orchestrator."""

from __future__ import annotations

import pytest

from cairn.models.events import Event, EventLog, EventType
from cairn.models.graph_types import GraphNode, IdeaGraph, NodeType
from cairn.models.workspace import WorkspaceRegistry
from cairn.pipeline.mutator import apply_event


class TestWorkspaceIdPropagation:
    def test_apply_event_sets_workspace_id(self):
        """Nodes created by apply_event receive the provided workspace_id."""
        graph = IdeaGraph()
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Test idea", "source": "user", "related_node_ids": []},
        )
        result = apply_event(graph, event, workspace_id="ws_abc")
        node_id = result.created_node_ids[0]
        node = graph.get_node(node_id)
        assert node.workspace_id == "ws_abc"

    def test_apply_event_default_workspace_id(self):
        """Default workspace_id is empty string (backward compat)."""
        graph = IdeaGraph()
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Test idea", "source": "user", "related_node_ids": []},
        )
        result = apply_event(graph, event)
        node_id = result.created_node_ids[0]
        node = graph.get_node(node_id)
        assert node.workspace_id == ""

    def test_replay_events_preserves_workspace_id(self):
        """replay_events reads workspace_id from events."""
        from cairn.pipeline.mutator import replay_events

        events = [
            Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": "Idea in ws1", "source": "user", "related_node_ids": [], "node_id": "ws1node"},
                workspace_id="ws1",
            ),
            Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": "Idea in ws2", "source": "user", "related_node_ids": [], "node_id": "ws2node"},
                workspace_id="ws2",
            ),
        ]
        graph = IdeaGraph()
        replay_events(graph, events)

        ws1_nodes = graph.get_nodes_by_workspace("ws1")
        ws2_nodes = graph.get_nodes_by_workspace("ws2")
        assert len(ws1_nodes) == 1
        assert len(ws2_nodes) == 1
        assert ws1_nodes[0].text == "Idea in ws1"
        assert ws2_nodes[0].text == "Idea in ws2"


class TestEventLogWorkspace:
    def test_append_and_retrieve_workspace_id(self):
        log = EventLog(":memory:")
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Idea", "source": "user", "related_node_ids": []},
            workspace_id="ws_test",
        )
        log.append(event)
        all_events = log.get_all()
        assert all_events[0].workspace_id == "ws_test"

    def test_get_by_workspace(self):
        log = EventLog(":memory:")
        log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Idea A", "source": "user", "related_node_ids": []},
            workspace_id="ws_a",
        ))
        log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Idea B", "source": "user", "related_node_ids": []},
            workspace_id="ws_b",
        ))
        ws_a_events = log.get_by_workspace("ws_a")
        assert len(ws_a_events) == 1
        assert ws_a_events[0].payload["text"] == "Idea A"

    def test_legacy_events_default_to_empty_workspace(self):
        """Events without workspace_id fall back to ''."""
        log = EventLog(":memory:")
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Legacy idea", "source": "user", "related_node_ids": []},
            # no workspace_id — defaults to ""
        )
        log.append(event)
        retrieved = log.get_all()
        assert retrieved[0].workspace_id == ""


class TestGraphWorkspaceFiltering:
    def test_get_nodes_by_workspace(self):
        from cairn.models.graph_types import IdeaGraph

        graph = IdeaGraph()
        n1 = GraphNode(type=NodeType.PROPOSITION, text="A", workspace_id="ws1")
        n2 = GraphNode(type=NodeType.PROPOSITION, text="B", workspace_id="ws2")
        n3 = GraphNode(type=NodeType.PROPOSITION, text="C", workspace_id="ws1")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        ws1_nodes = graph.get_nodes_by_workspace("ws1")
        assert len(ws1_nodes) == 2
        texts = {n.text for n in ws1_nodes}
        assert texts == {"A", "C"}

    def test_node_summary_list_workspace_filter(self):
        from cairn.models.graph_types import IdeaGraph

        graph = IdeaGraph()
        n1 = GraphNode(type=NodeType.PROPOSITION, text="A", workspace_id="ws1")
        n2 = GraphNode(type=NodeType.PROPOSITION, text="B", workspace_id="ws2")
        graph.add_node(n1)
        graph.add_node(n2)

        summary = graph.node_summary_list(workspace_id="ws1")
        assert len(summary) == 1
        assert summary[0]["text"] == "A"

    def test_node_summary_list_no_filter(self):
        from cairn.models.graph_types import IdeaGraph

        graph = IdeaGraph()
        n1 = GraphNode(type=NodeType.PROPOSITION, text="A", workspace_id="ws1")
        n2 = GraphNode(type=NodeType.PROPOSITION, text="B", workspace_id="ws2")
        graph.add_node(n1)
        graph.add_node(n2)

        summary = graph.node_summary_list()
        assert len(summary) == 2


class TestOrchestratorWorkspaceWiring:
    @pytest.mark.skip(reason="cairn.pipeline.orchestrator not yet implemented")
    async def test_nodes_tagged_with_active_workspace(self):
        """Nodes created during a mocked turn get the active workspace_id."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cairn.models.events import EventLog
        from cairn.pipeline.orchestrator import Orchestrator

        event_log = EventLog(":memory:")
        workspace_registry = WorkspaceRegistry(":memory:")
        graph = IdeaGraph()

        orchestrator = Orchestrator(
            event_log=event_log,
            graph=graph,
            workspace_registry=workspace_registry,
            embedding_store=None,  # no embeddings for this test
            active_workspace_id="default",
        )

        # Mock classify_exchange to return a single NEW_PROPOSITION event
        classified = MagicMock()
        classified.event_type = EventType.NEW_PROPOSITION
        classified.payload = {
            "text": "Test proposition",
            "source": "user",
            "related_node_ids": [],
        }
        classified.reasoning = "test"

        # Mock evaluate_next_move
        eval_output = MagicMock()
        eval_output.selected_move = MagicMock(value="probe")
        eval_output.reasoning = "test"

        with (
            patch("cairn.pipeline.orchestrator.classify_exchange", new=AsyncMock(return_value=[classified])),
            patch("cairn.pipeline.orchestrator.evaluate_next_move", new=AsyncMock(return_value=eval_output)),
        ):
            result = await orchestrator.prepare_turn("I have an idea about testing")

        # Verify node was created with the active workspace_id
        assert len(result.applied_events) == 1
        applied_event = result.applied_events[0]
        assert applied_event.workspace_id == "default"

        # Verify the graph node has workspace_id
        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == 1
        assert all_nodes[0].workspace_id == "default"
