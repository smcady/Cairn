"""Tests for the cairn CLI init command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cairn.cli import (
    _check_api_keys,
    _configure_hooks,
    _configure_mcp,
    _find_cairn_root,
    _load_json,
    _write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_cairn_root(tmp_path: Path) -> Path:
    """Create a minimal fake Cairn repo structure."""
    root = tmp_path / "cairn-install"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "hook_ingest.py").touch()
    (root / "scripts" / "hook_orient.py").touch()
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").touch()
    return root


# ---------------------------------------------------------------------------
# Path detection
# ---------------------------------------------------------------------------

class TestFindCairnRoot:
    def test_finds_root_from_package(self):
        """Should find the repo root from cairn.__file__."""
        root = _find_cairn_root()
        assert (root / "scripts" / "hook_ingest.py").exists()
        assert (root / "scripts" / "hook_orient.py").exists()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_load_missing_file(self, tmp_path):
        assert _load_json(tmp_path / "nope.json") == {}

    def test_load_malformed_file(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        assert _load_json(bad) == {}

    def test_round_trip(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "nested": {"a": 1}}
        _write_json(path, data)
        assert _load_json(path) == data

    def test_write_creates_parents(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.json"
        _write_json(path, {"x": 1})
        assert path.exists()


# ---------------------------------------------------------------------------
# Hook configuration
# ---------------------------------------------------------------------------

class TestConfigureHooks:
    def test_creates_settings_from_scratch(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        _configure_hooks(project, venv_python, root, db_path)

        settings = _load_json(project / ".claude" / "settings.json")
        assert "hooks" in settings
        assert "Stop" in settings["hooks"]
        assert "UserPromptSubmit" in settings["hooks"]

        # Verify hook commands contain absolute paths
        stop_cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        assert str(venv_python) in stop_cmd
        assert "hook_ingest.py" in stop_cmd
        assert str(db_path) in stop_cmd

        orient_cmd = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert str(venv_python) in orient_cmd
        assert "hook_orient.py" in orient_cmd
        assert settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["timeout"] == 10000

    def test_preserves_existing_settings(self, tmp_path):
        project = tmp_path / "project"
        (project / ".claude").mkdir(parents=True)
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        # Write existing settings with other hooks
        existing = {
            "hooks": {
                "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "echo other"}]}]
            },
            "other_setting": True,
        }
        _write_json(project / ".claude" / "settings.json", existing)

        _configure_hooks(project, venv_python, root, db_path)

        settings = _load_json(project / ".claude" / "settings.json")
        # Other setting preserved
        assert settings["other_setting"] is True
        # Other hook preserved
        assert len(settings["hooks"]["Stop"]) == 2
        # Cairn hook added
        assert any("hook_ingest.py" in h["hooks"][0]["command"] for h in settings["hooks"]["Stop"])

    def test_quotes_paths_with_spaces(self, tmp_path):
        """Paths containing spaces are properly quoted in hook commands."""
        project = tmp_path / "my project"
        project.mkdir()
        root_dir = tmp_path / "cairn install"
        (root_dir / "scripts").mkdir(parents=True)
        (root_dir / "scripts" / "hook_ingest.py").touch()
        (root_dir / "scripts" / "hook_orient.py").touch()
        (root_dir / ".venv" / "bin").mkdir(parents=True)
        (root_dir / ".venv" / "bin" / "python").touch()
        venv_python = root_dir / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        _configure_hooks(project, venv_python, root_dir, db_path)

        settings = _load_json(project / ".claude" / "settings.json")
        stop_cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        # Python path and script path should be quoted
        assert f'"{venv_python}"' in stop_cmd
        assert '"' + str(root_dir / "scripts" / "hook_ingest.py") + '"' in stop_cmd

    def test_idempotent(self, tmp_path):
        """Running twice doesn't duplicate hooks."""
        project = tmp_path / "project"
        project.mkdir()
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        _configure_hooks(project, venv_python, root, db_path)
        _configure_hooks(project, venv_python, root, db_path)

        settings = _load_json(project / ".claude" / "settings.json")
        assert len(settings["hooks"]["Stop"]) == 1
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1


# ---------------------------------------------------------------------------
# MCP configuration
# ---------------------------------------------------------------------------

class TestConfigureMcp:
    def test_creates_mcp_from_scratch(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        _configure_mcp(project, venv_python, root, db_path)

        config = _load_json(project / ".mcp.json")
        assert "cairn" in config["mcpServers"]
        server = config["mcpServers"]["cairn"]
        assert server["command"] == str(venv_python)
        assert server["args"] == ["-m", "cairn.mcp_server"]
        assert server["cwd"] == str(root)
        assert server["env"]["CAIRN_DB"] == str(db_path)

    def test_preserves_other_servers(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        # Write existing MCP config with another server
        existing = {"mcpServers": {"other-tool": {"command": "other"}}}
        _write_json(project / ".mcp.json", existing)

        _configure_mcp(project, venv_python, root, db_path)

        config = _load_json(project / ".mcp.json")
        assert "other-tool" in config["mcpServers"]
        assert "cairn" in config["mcpServers"]

    def test_idempotent(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        root = _fake_cairn_root(tmp_path)
        venv_python = root / ".venv" / "bin" / "python"
        db_path = project / "cairn.db"

        _configure_mcp(project, venv_python, root, db_path)
        _configure_mcp(project, venv_python, root, db_path)

        config = _load_json(project / ".mcp.json")
        assert len(config["mcpServers"]) == 1


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

class TestCheckApiKeys:
    def test_returns_fastembed_when_no_voyage(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        result = _check_api_keys(tmp_path)
        assert "fastembed" in result

    def test_returns_voyage_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("VOYAGE_API_KEY", "test")
        result = _check_api_keys(tmp_path)
        assert "Voyage" in result

    def test_warns_on_missing_anthropic(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        _check_api_keys(tmp_path)
        output = capsys.readouterr().out
        assert "ANTHROPIC_API_KEY" in output
