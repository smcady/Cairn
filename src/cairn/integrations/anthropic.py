"""Drop-in AsyncAnthropic wrapper that auto-ingests every exchange into Cairn.

Usage:
    # Before
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()

    # After — same interface, auto-ingests every exchange
    from cairn.integrations.anthropic import AsyncAnthropic
    client = AsyncAnthropic()  # or AsyncAnthropic(cairn_db="path/to/db")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from cairn._engine_registry import get_engine

logger = logging.getLogger("cairn")

# Keep a reference to background tasks so they aren't garbage collected
_background_tasks: set[asyncio.Task[None]] = set()


def _extract_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message text from the messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
    return ""


def _extract_assistant_text(response: anthropic.types.Message) -> str:
    """Extract text from an Anthropic Message response."""
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return " ".join(parts)


async def _ingest_exchange(
    user_text: str,
    assistant_text: str,
    db_path: str | None,
) -> None:
    """Ingest a user+assistant exchange into the reasoning graph. Never raises."""
    try:
        if not user_text and not assistant_text:
            return

        parts = []
        if user_text:
            parts.append(f"User: {user_text}")
        if assistant_text:
            parts.append(f"Assistant: {assistant_text}")
        content = "\n\n".join(parts)

        engine = get_engine(db_path)
        await engine.ingest(content, source="sdk")
    except Exception:
        logger.debug("cairn: ingest failed", exc_info=True)


def _schedule_ingest(
    user_text: str,
    assistant_text: str,
    db_path: str | None,
) -> None:
    """Fire-and-forget ingest as a background task."""
    task = asyncio.create_task(
        _ingest_exchange(user_text, assistant_text, db_path)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class _CairnMessages:
    """Wraps anthropic.resources.AsyncMessages to intercept create() and stream()."""

    def __init__(
        self,
        inner: anthropic.resources.AsyncMessages,
        db_path: str | None,
    ) -> None:
        self._inner = inner
        self._db_path = db_path

    async def create(self, **kwargs: Any) -> anthropic.types.Message:
        """Call the original create() and schedule background ingest."""
        response = await self._inner.create(**kwargs)

        try:
            messages = kwargs.get("messages", [])
            user_text = _extract_user_text(messages)
            assistant_text = _extract_assistant_text(response)
            _schedule_ingest(user_text, assistant_text, self._db_path)
        except Exception:
            logger.debug("cairn: failed to schedule ingest", exc_info=True)

        return response

    def stream(self, **kwargs: Any) -> _CairnStreamManager:
        """Return a wrapped stream that ingests after completion."""
        return _CairnStreamManager(
            inner_stream_fn=self._inner.stream,
            kwargs=kwargs,
            db_path=self._db_path,
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the inner messages resource."""
        return getattr(self._inner, name)


class _CairnStreamManager:
    """Async context manager wrapping the Anthropic stream to ingest on completion."""

    def __init__(
        self,
        inner_stream_fn: Any,
        kwargs: dict[str, Any],
        db_path: str | None,
    ) -> None:
        self._inner_stream_fn = inner_stream_fn
        self._kwargs = kwargs
        self._db_path = db_path
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._stream = self._inner_stream_fn(**self._kwargs)
        return await self._stream.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        stream = self._stream
        try:
            await stream.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if exc_type is None:
                try:
                    final_message = stream.get_final_message()
                    messages = self._kwargs.get("messages", [])
                    user_text = _extract_user_text(messages)
                    assistant_text = _extract_assistant_text(final_message)
                    _schedule_ingest(user_text, assistant_text, self._db_path)
                except Exception:
                    logger.debug("cairn: stream ingest failed", exc_info=True)


class AsyncAnthropic(anthropic.AsyncAnthropic):
    """Drop-in replacement for anthropic.AsyncAnthropic with Cairn auto-ingest.

    Accepts all the same arguments as the original, plus an optional
    ``cairn_db`` keyword argument specifying the database path. If omitted,
    falls back to ``cairn.init()`` configuration or the ``CAIRN_DB``
    environment variable.
    """

    def __init__(self, *, cairn_db: str | None = None, **kwargs: Any) -> None:
        self._cairn_db = cairn_db
        super().__init__(**kwargs)

    @property  # type: ignore[override]
    def messages(self) -> _CairnMessages:  # type: ignore[override]
        """Override the messages property to return our wrapped version."""
        # Cache the wrapper on the instance to avoid re-creating it
        if not hasattr(self, "_cairn_messages"):
            inner = anthropic.resources.AsyncMessages(self)
            self._cairn_messages = _CairnMessages(inner, self._cairn_db)
        return self._cairn_messages
