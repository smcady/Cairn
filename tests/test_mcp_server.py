"""Unit tests for cairn.mcp_server — incremental rebuild and health checks.

Tests the _get_engine() incremental path and harness_status warnings
without making real API calls.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from cairn.models.events import Event, EventLog, EventType
from cairn.models.graph_types import IdeaGraph, NodeType
from cairn.pipeline.mutator import apply_event, replay_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, payload: dict, turn: int = 1) -> Event:
    """Create a minimal Event for testing."""
    return Event(event_type=event_type, payload=payload, turn_number=turn)


def _seed_events(event_log: EventLog, count: int = 3) -> list[Event]:
    """Write `count` proposition events to an event log and return them."""
    events = []
    for i in range(count):
        e = event_log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": f"Proposition {i}", "node_id": f"node_{i:04d}"},
            turn_number=i + 1,
        ))
        events.append(e)
    return events


# ---------------------------------------------------------------------------
# _get_engine() — incremental rebuild
# ---------------------------------------------------------------------------

class TestIncrementalRebuild:
    """Test that _get_engine() applies only new events after initial load."""

    def test_initial_load_builds_full_graph(self, tmp_path):
        """First call to _get_engine builds graph from all events."""
        db = str(tmp_path / "test.db")
        log = EventLog(db)
        events = _seed_events(log, 3)

        # Build graph the same way _get_engine does
        graph = IdeaGraph()
        replay_events(graph, log.get_all())

        assert graph.node_count() == 3

    def test_incremental_applies_only_new_events(self, tmp_path):
        """After initial load, only new events should be applied."""
        db = str(tmp_path / "test.db")
        log = EventLog(db)

        # Initial full load
        initial = _seed_events(log, 2)
        graph = IdeaGraph()
        replay_events(graph, log.get_all())
        assert graph.node_count() == 2

        last_id = initial[-1].id

        # Add more events (simulating Stop hook writing to DB)
        log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "New proposition", "node_id": "node_new1"},
            turn_number=3,
        ))

        # Incremental: only get events since last_id
        new_events = log.get_since(last_id)
        assert len(new_events) == 1
        assert new_events[0].payload["text"] == "New proposition"

        # Apply incrementally (no clear!)
        for e in new_events:
            apply_event(graph, e, workspace_id=e.workspace_id)
        assert graph.node_count() == 3

    def test_no_new_events_is_noop(self, tmp_path):
        """When no new events exist, get_since returns empty list."""
        db = str(tmp_path / "test.db")
        log = EventLog(db)
        events = _seed_events(log, 2)
        last_id = events[-1].id

        new = log.get_since(last_id)
        assert new == []

    def test_multiple_incremental_rounds(self, tmp_path):
        """Simulate multiple conversation captures with incremental updates."""
        db = str(tmp_path / "test.db")
        log = EventLog(db)

        # Round 1: initial full load
        _seed_events(log, 2)
        graph = IdeaGraph()
        all_events = log.get_all()
        replay_events(graph, all_events)
        last_id = all_events[-1].id
        assert graph.node_count() == 2

        # Round 2: incremental
        log.append(Event(
            event_type=EventType.NEW_PROPOSITION,
            payload={"text": "Round 2 idea", "node_id": "node_r2"},
            turn_number=3,
        ))
        new = log.get_since(last_id)
        for e in new:
            apply_event(graph, e, workspace_id=e.workspace_id)
        last_id = new[-1].id
        assert graph.node_count() == 3

        # Round 3: incremental
        log.append(Event(
            event_type=EventType.CONTRADICTION,
            payload={
                "objection_text": "Round 2 idea is wrong",
                "node_id": "node_contra",
                "target_node_id": "node_r2",
            },
            turn_number=4,
        ))
        new = log.get_since(last_id)
        for e in new:
            apply_event(graph, e, workspace_id=e.workspace_id)
        assert graph.node_count() == 4


# ---------------------------------------------------------------------------
# harness_status — health check warnings
# ---------------------------------------------------------------------------

class TestStatusHealthChecks:
    """Test that harness_status surfaces configuration problems."""

    def _reset_server(self, db: str):
        """Reset MCP server module state for testing."""
        import cairn.mcp_server as server
        server._engine = None
        server._last_event_id = 0
        server.DB_PATH = db
        return server

    @patch("cairn.mcp_server.VectorIndex", return_value=MagicMock())
    def test_warns_on_empty_graph(self, mock_vi, tmp_path):
        """Empty graph should produce a warning about Stop hook."""
        db = str(tmp_path / "test.db")

        with patch.dict(os.environ, {
            "CAIRN_DB": db,
            "ANTHROPIC_API_KEY": "test-key",
            "VOYAGE_API_KEY": "test-key",
        }):
            server = self._reset_server(db)
            result = server.status()
            assert "No events ingested" in result or "empty" in result.lower()

    @patch("cairn.mcp_server.VectorIndex", return_value=MagicMock())
    def test_warns_on_missing_anthropic_key(self, mock_vi, tmp_path):
        """Missing ANTHROPIC_API_KEY should produce a warning."""
        db = str(tmp_path / "test.db")

        clean_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        clean_env.update({"CAIRN_DB": db, "VOYAGE_API_KEY": "test-key"})

        with patch.dict(os.environ, clean_env, clear=True):
            server = self._reset_server(db)
            result = server.status()
            assert "ANTHROPIC_API_KEY" in result

    @patch("cairn.mcp_server.VectorIndex", return_value=MagicMock())
    def test_warns_on_missing_voyage_key(self, mock_vi, tmp_path):
        """Missing VOYAGE_API_KEY should produce a warning."""
        db = str(tmp_path / "test.db")

        clean_env = {k: v for k, v in os.environ.items() if k != "VOYAGE_API_KEY"}
        clean_env.update({"CAIRN_DB": db, "ANTHROPIC_API_KEY": "test-key"})

        with patch.dict(os.environ, clean_env, clear=True):
            server = self._reset_server(db)
            result = server.status()
            assert "VOYAGE_API_KEY" in result

    @patch("cairn.mcp_server.VectorIndex", return_value=MagicMock())
    def test_no_warnings_when_healthy(self, mock_vi, tmp_path):
        """A populated graph with all keys set should produce no warnings."""
        db = str(tmp_path / "test.db")

        # Seed some events
        log = EventLog(db)
        _seed_events(log, 2)

        with patch.dict(os.environ, {
            "CAIRN_DB": db,
            "ANTHROPIC_API_KEY": "test-key",
            "VOYAGE_API_KEY": "test-key",
        }):
            server = self._reset_server(db)
            result = server.status()
            assert "Warnings" not in result


# ---------------------------------------------------------------------------
# hook_orient.py — basic smoke test
# ---------------------------------------------------------------------------

class TestHookOrient:
    """Test the UserPromptSubmit orient hook script logic."""

    def test_empty_prompt_exits_silently(self):
        """Empty prompt should produce no output."""
        import json
        from io import StringIO
        from unittest.mock import patch as mock_patch

        stdin_data = json.dumps({"prompt": ""})

        with mock_patch("sys.stdin", StringIO(stdin_data)):
            from scripts.hook_orient import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_invalid_json_exits_silently(self):
        """Invalid JSON on stdin should exit 0, not crash."""
        from io import StringIO
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.stdin", StringIO("not json")):
            from scripts.hook_orient import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
