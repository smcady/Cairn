"""Tests for EventLog persistence and replay."""

import pytest

from cairn.models.events import (
    Event,
    EventLog,
    EventType,
    NewPropositionPayload,
    SupportPayload,
    _compute_event_hash,
)
from cairn.models.graph_types import IdeaGraph, NodeType
from cairn.pipeline.mutator import apply_event, replay_events


@pytest.fixture
def event_log():
    log = EventLog(":memory:")
    yield log
    log.close()


class TestEventLog:
    def test_append_and_retrieve(self, event_log):
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Test idea", "source": "user", "related_node_ids": []},
            turn_number=1,
            session_id="test123",
        )
        result = event_log.append(event)
        assert result.id is not None
        assert result.id == 1

    def test_get_all(self, event_log):
        for i in range(3):
            event_log.append(Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": f"Idea {i}", "source": "user", "related_node_ids": []},
                turn_number=i + 1,
                session_id="test123",
            ))

        all_events = event_log.get_all()
        assert len(all_events) == 3
        assert all_events[0].payload["text"] == "Idea 0"
        assert all_events[2].payload["text"] == "Idea 2"

    def test_get_since(self, event_log):
        for i in range(5):
            event_log.append(Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": f"Idea {i}", "source": "user", "related_node_ids": []},
                turn_number=i + 1,
                session_id="test123",
            ))

        since_3 = event_log.get_since(3)
        assert len(since_3) == 2
        assert since_3[0].id == 4
        assert since_3[1].id == 5

    def test_get_by_session(self, event_log):
        event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Session A", "source": "user", "related_node_ids": []},
            session_id="session_a",
        ))
        event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Session B", "source": "user", "related_node_ids": []},
            session_id="session_b",
        ))

        a_events = event_log.get_by_session("session_a")
        assert len(a_events) == 1
        assert a_events[0].payload["text"] == "Session A"

    def test_get_recent(self, event_log):
        for i in range(10):
            event_log.append(Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": f"Idea {i}", "source": "user", "related_node_ids": []},
                turn_number=i + 1,
                session_id="test123",
            ))

        recent = event_log.get_recent(3)
        assert len(recent) == 3
        # Should be in chronological order
        assert recent[0].payload["text"] == "Idea 7"
        assert recent[2].payload["text"] == "Idea 9"

    def test_count(self, event_log):
        assert event_log.count() == 0
        event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Idea", "source": "user", "related_node_ids": []},
            session_id="test",
        ))
        assert event_log.count() == 1

    def test_event_type_preserved(self, event_log):
        event_log.append(Event(
            event_type=EventType.CONTRADICTION,
            payload={"target_node_id": "n1", "objection_text": "Disagree", "source": "user"},
            session_id="test",
        ))

        retrieved = event_log.get_all()
        assert retrieved[0].event_type == EventType.CONTRADICTION

    def test_get_typed_payload(self):
        event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Test idea", "source": "user", "related_node_ids": ["n1"]},
        )
        typed = event.get_typed_payload()
        assert isinstance(typed, NewPropositionPayload)
        assert typed.text == "Test idea"
        assert typed.related_node_ids == ["n1"]


class TestHashChain:
    def test_first_event_has_empty_parent_hash(self, event_log):
        e = event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "First", "source": "user", "related_node_ids": []},
        ))
        assert e.parent_hash == ""

    def test_second_event_hash_links_to_first(self, event_log):
        e1 = event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "First", "source": "user", "related_node_ids": []},
        ))
        e2 = event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Second", "source": "user", "related_node_ids": []},
        ))
        assert e2.parent_hash == _compute_event_hash(e1)
        assert e2.parent_hash != ""

    def test_verify_chain_intact(self, event_log):
        for i in range(5):
            event_log.append(Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": f"Idea {i}", "source": "user", "related_node_ids": []},
            ))
        ok, msg = event_log.verify_chain()
        assert ok, msg

    def test_verify_chain_detects_tampering(self, event_log):
        for i in range(3):
            event_log.append(Event(
                event_type=EventType.NEW_PROPOSITION,
                payload={"text": f"Idea {i}", "source": "user", "related_node_ids": []},
            ))
        # Tamper directly in SQLite
        event_log._conn.execute("UPDATE events SET payload = '{\"text\": \"TAMPERED\", \"source\": \"user\", \"related_node_ids\": []}' WHERE id = 1")
        event_log._conn.commit()
        ok, msg = event_log.verify_chain()
        assert not ok

    def test_parent_hash_persists_and_reloads(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        log1 = EventLog(db_path)
        e1 = log1.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Persistent", "source": "user", "related_node_ids": []},
        ))
        expected_hash = _compute_event_hash(e1)
        log1.close()

        log2 = EventLog(db_path)
        e2 = log2.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "After reload", "source": "user", "related_node_ids": []},
        ))
        assert e2.parent_hash == expected_hash
        ok, msg = log2.verify_chain()
        assert ok, msg
        log2.close()


class TestEventReplay:
    def test_replay_rebuilds_graph(self, event_log):
        # Create a sequence of events
        event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "First idea", "source": "user", "related_node_ids": []},
            turn_number=1,
            session_id="test",
        ))

        # Build graph from events
        graph = IdeaGraph()
        events = event_log.get_all()
        replay_events(graph, events)

        assert graph.node_count() == 1
        nodes = graph.get_all_nodes(NodeType.PROPOSITION)
        assert len(nodes) == 1
        assert nodes[0].text == "First idea"

    def test_replay_with_support(self):
        graph = IdeaGraph()

        # Manually apply events in sequence
        prop_event = Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Claim X", "source": "user", "related_node_ids": [], "node_id": "prop_claimx"},
            turn_number=1,
            session_id="test",
        )
        result = apply_event(graph, prop_event)
        prop_id = result.created_node_ids[0]

        support_event = Event(
            event_type=EventType.SUPPORT,
            payload={"target_node_id": prop_id, "evidence_text": "Because Y", "source": "user", "evidence_node_id": "evid_y"},
            turn_number=2,
            session_id="test",
        )
        apply_event(graph, support_event)

        # Verify
        prop = graph.get_node(prop_id)
        assert prop.confidence == 0.6  # 0.5 + 0.1
        assert graph.node_count() == 2  # proposition + evidence

    def test_replay_idempotent(self, event_log):
        """Replaying the same events twice should produce the same graph."""
        event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Idea A", "source": "user", "related_node_ids": []},
            turn_number=1,
            session_id="test",
        ))

        events = event_log.get_all()

        graph1 = IdeaGraph()
        replay_events(graph1, events)

        graph2 = IdeaGraph()
        replay_events(graph2, events)

        assert graph1.node_count() == graph2.node_count()
        assert graph1.edge_count() == graph2.edge_count()
