"""Tests for the Cairn SDK wrapper (engine registry + AsyncAnthropic wrapper)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cairn._engine_registry import _resolve_db_path, get_engine, init, reset
from cairn.integrations.anthropic import (
    AsyncAnthropic,
    _CairnMessages,
    _extract_assistant_text,
    _extract_user_text,
)


# ---------------------------------------------------------------------------
# Engine registry tests
# ---------------------------------------------------------------------------


class TestResolveDbPath:
    def setup_method(self):
        reset()

    def teardown_method(self):
        reset()

    def test_explicit_path_wins(self, tmp_path):
        db = str(tmp_path / "explicit.db")
        path = _resolve_db_path(db)
        assert path == db

    def test_init_default_used(self, tmp_path):
        db = str(tmp_path / "init-default.db")
        init(db_path=db)
        path = _resolve_db_path()
        assert path == db

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        db = str(tmp_path / "env.db")
        monkeypatch.setenv("CAIRN_DB", db)
        path = _resolve_db_path()
        assert path == db

    def test_explicit_overrides_init(self, tmp_path):
        init(db_path=str(tmp_path / "init.db"))
        explicit = str(tmp_path / "explicit.db")
        path = _resolve_db_path(explicit)
        assert path == explicit

    def test_default_is_cairn_db_in_cwd(self):
        path = _resolve_db_path()
        assert path.endswith("cairn.db")


class TestGetEngine:
    def setup_method(self):
        reset()

    def teardown_method(self):
        reset()

    @pytest.fixture(autouse=True)
    def mock_voyage(self, monkeypatch):
        """Prevent VectorIndex from requiring a real Voyage API key."""
        monkeypatch.setenv("VOYAGE_API_KEY", "test-fake-key")

    def test_creates_engine(self, tmp_path):
        db = str(tmp_path / "test.db")
        engine = get_engine(db)
        assert engine is not None

    def test_returns_same_instance_for_same_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        e1 = get_engine(db)
        e2 = get_engine(db)
        assert e1 is e2

    def test_different_paths_get_different_engines(self, tmp_path):
        db1 = str(tmp_path / "a.db")
        db2 = str(tmp_path / "b.db")
        e1 = get_engine(db1)
        e2 = get_engine(db2)
        assert e1 is not e2

    def test_reset_clears_cache(self, tmp_path):
        db = str(tmp_path / "test.db")
        e1 = get_engine(db)
        reset()
        e2 = get_engine(db)
        assert e1 is not e2


# ---------------------------------------------------------------------------
# Helper extraction tests
# ---------------------------------------------------------------------------


class TestExtractUserText:
    def test_simple_string_content(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert _extract_user_text(messages) == "Hello"

    def test_block_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        assert _extract_user_text(messages) == "First second"

    def test_returns_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_user_text(messages) == "Second"

    def test_empty_messages(self):
        assert _extract_user_text([]) == ""


class TestExtractAssistantText:
    def test_extracts_text_blocks(self):
        msg = MagicMock()
        block1 = MagicMock(type="text", text="Hello")
        block2 = MagicMock(type="text", text="world")
        msg.content = [block1, block2]
        assert _extract_assistant_text(msg) == "Hello world"

    def test_skips_non_text_blocks(self):
        msg = MagicMock()
        text_block = MagicMock(type="text", text="Hello")
        tool_block = MagicMock(type="tool_use", text="ignored")
        msg.content = [text_block, tool_block]
        assert _extract_assistant_text(msg) == "Hello"


# ---------------------------------------------------------------------------
# AsyncAnthropic wrapper tests
# ---------------------------------------------------------------------------


class TestAsyncAnthropic:
    @pytest.fixture(autouse=True)
    def fake_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    def test_is_subclass(self):
        import anthropic

        assert issubclass(AsyncAnthropic, anthropic.AsyncAnthropic)

    def test_accepts_cairn_db(self):
        client = AsyncAnthropic(cairn_db="/tmp/test.db")
        assert client._cairn_db == "/tmp/test.db"

    def test_messages_returns_cairn_wrapper(self):
        client = AsyncAnthropic()
        assert isinstance(client.messages, _CairnMessages)

    def test_messages_is_cached(self):
        client = AsyncAnthropic()
        m1 = client.messages
        m2 = client.messages
        assert m1 is m2


class TestCairnMessagesCreate:
    @pytest.fixture
    def mock_inner(self):
        inner = AsyncMock()
        response = MagicMock()
        text_block = MagicMock(type="text", text="Assistant reply")
        response.content = [text_block]
        inner.create.return_value = response
        return inner

    async def test_calls_inner_create(self, mock_inner):
        msgs = _CairnMessages(mock_inner, db_path=None)
        messages = [{"role": "user", "content": "Hello"}]

        with patch("cairn.integrations.anthropic._schedule_ingest"):
            result = await msgs.create(messages=messages, model="test", max_tokens=100)

        mock_inner.create.assert_awaited_once_with(
            messages=messages, model="test", max_tokens=100
        )
        assert result is mock_inner.create.return_value

    async def test_schedules_ingest(self, mock_inner):
        msgs = _CairnMessages(mock_inner, db_path="/tmp/test.db")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("cairn.integrations.anthropic._schedule_ingest") as mock_sched:
            await msgs.create(messages=messages, model="test", max_tokens=100)

        mock_sched.assert_called_once_with("Hello", "Assistant reply", "/tmp/test.db")

    async def test_returns_response_even_if_ingest_errors(self, mock_inner):
        msgs = _CairnMessages(mock_inner, db_path=None)
        messages = [{"role": "user", "content": "Hello"}]

        with patch(
            "cairn.integrations.anthropic._schedule_ingest",
            side_effect=RuntimeError("boom"),
        ):
            result = await msgs.create(messages=messages, model="test", max_tokens=100)
            assert result is mock_inner.create.return_value


class TestIngestExchange:
    async def test_ingest_formats_content(self):
        from cairn.integrations.anthropic import _ingest_exchange

        with patch("cairn.integrations.anthropic.get_engine") as mock_get:
            mock_engine = AsyncMock()
            mock_get.return_value = mock_engine
            await _ingest_exchange("Hello user", "Hello assistant", "/tmp/db")

        mock_engine.ingest.assert_awaited_once_with(
            "User: Hello user\n\nAssistant: Hello assistant",
            source="sdk",
        )

    async def test_ingest_skips_empty(self):
        from cairn.integrations.anthropic import _ingest_exchange

        with patch("cairn.integrations.anthropic.get_engine") as mock_get:
            await _ingest_exchange("", "", None)

        mock_get.assert_not_called()

    async def test_ingest_swallows_errors(self):
        from cairn.integrations.anthropic import _ingest_exchange

        with patch(
            "cairn.integrations.anthropic.get_engine",
            side_effect=RuntimeError("db error"),
        ):
            await _ingest_exchange("Hello", "World", None)
