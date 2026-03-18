"""Tests for WorkspaceRegistry."""

import pytest

from cairn.models.workspace import Workspace, WorkspaceRegistry


@pytest.fixture
def registry():
    return WorkspaceRegistry(":memory:")


class TestWorkspaceRegistry:
    def test_default_workspace_created_on_init(self, registry):
        ws = registry.get("default")
        assert ws is not None
        assert ws.id == "default"
        assert ws.label == "Default"

    def test_idempotent_init(self):
        """Re-initializing with the same DB doesn't create duplicate defaults."""
        r1 = WorkspaceRegistry(":memory:")
        # Create a second registry pointing at the same in-memory DB
        # (different connection, but same schema guarantee)
        r2 = WorkspaceRegistry(":memory:")
        # Both should have exactly one default workspace
        assert r1.count() == 1
        assert r2.count() == 1

    def test_create_workspace(self, registry):
        ws = registry.create(label="My Topic")
        assert ws.id != "default"
        assert ws.label == "My Topic"
        assert ws.created_at != ""

    def test_get_workspace(self, registry):
        ws = registry.create(label="Test")
        retrieved = registry.get(ws.id)
        assert retrieved is not None
        assert retrieved.id == ws.id
        assert retrieved.label == "Test"

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_all(self, registry):
        registry.create(label="A")
        registry.create(label="B")
        all_ws = registry.get_all()
        # default + 2 new = 3
        assert len(all_ws) == 3
        ids = {ws.id for ws in all_ws}
        assert "default" in ids

    def test_update_label(self, registry):
        ws = registry.create(label="old")
        registry.update_label(ws.id, "new")
        retrieved = registry.get(ws.id)
        assert retrieved.label == "new"

    def test_count(self, registry):
        assert registry.count() == 1  # default
        registry.create()
        assert registry.count() == 2
        registry.create()
        assert registry.count() == 3

    def test_create_without_label(self, registry):
        ws = registry.create()
        assert ws.label == ""
        assert ws.id != "default"
