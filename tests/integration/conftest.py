"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent

# Make conversation_loader importable
sys.path.insert(0, str(ROOT / "tests" / "integration" / "external_project"))


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database path and clean up after."""
    db_path = str(tmp_path / "test.db")
    yield db_path
    for suffix in ["", "-shm", "-wal"]:
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink()


@pytest.fixture
def reset_cairn_registry():
    """Reset the engine registry between tests."""
    from cairn._engine_registry import _engines
    _engines.clear()
    yield
    _engines.clear()
