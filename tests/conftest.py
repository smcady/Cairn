"""Pytest configuration and session-level fixtures.

Provides:
- asyncio mode configuration
- API usage metrics summary printed after each test session

Usage in integration tests:
    async def test_something(self, metrics_recorder):
        engine = _make_engine()
        # ... run test ...
        metrics_recorder(engine)   # records engine's LLM + embed usage
"""

from __future__ import annotations

import pytest

from cairn.utils.metrics import SessionMetrics


# ---------------------------------------------------------------------------
# Session-level metrics accumulator
# ---------------------------------------------------------------------------

_session_metrics = SessionMetrics()
_integration_tests_ran = 0


def _merge_engine_metrics(engine) -> None:
    """Read metrics from a MemoryEngine instance into the session accumulator."""
    engine.get_stats()  # syncs vector_index._request_count → engine.metrics.embed_calls
    m = engine.metrics
    _session_metrics.llm_calls += m.llm_calls
    _session_metrics.llm_input_tokens += m.llm_input_tokens
    _session_metrics.llm_output_tokens += m.llm_output_tokens
    _session_metrics.llm_cache_read_tokens += m.llm_cache_read_tokens
    _session_metrics.llm_cache_write_tokens += m.llm_cache_write_tokens
    _session_metrics.embed_calls += m.embed_calls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that make real API calls (skip with -m 'not integration')",
    )


@pytest.fixture(scope="session")
def metrics_recorder():
    """Session-scoped fixture: returns a callable that records a MemoryEngine's API usage.

    Call it after each engine finishes work so its tokens are counted in the summary.

    Example:
        async def test_something(metrics_recorder):
            engine = _make_engine()
            await engine.ingest("some text")
            metrics_recorder(engine)
    """
    return _merge_engine_metrics


@pytest.fixture(autouse=True)
def _count_integration_tests(request):
    global _integration_tests_ran
    if request.node.get_closest_marker("integration"):
        _integration_tests_ran += 1
    yield


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print API usage at the end of the test session."""
    has_llm = _session_metrics.llm_calls > 0
    has_embed = _session_metrics.embed_calls > 0

    if not has_llm and not has_embed:
        if _integration_tests_ran > 0:
            terminalreporter.write_sep("-", "API Usage")
            terminalreporter.write_line(
                f"  {_integration_tests_ran} integration test(s) ran but no API usage was recorded."
            )
            terminalreporter.write_line(
                "  Use the `metrics_recorder(engine)` fixture to capture token counts."
            )
        return

    total_tokens = _session_metrics.llm_total_tokens
    terminalreporter.write_sep("-", "API Usage")
    terminalreporter.write_line(f"  LLM calls:     {_session_metrics.llm_calls}")
    terminalreporter.write_line(f"  Input tokens:  {_session_metrics.llm_input_tokens:,}")
    terminalreporter.write_line(f"  Output tokens: {_session_metrics.llm_output_tokens:,}")
    terminalreporter.write_line(f"  Total tokens:  {total_tokens:,}")
    if _session_metrics.llm_cache_read_tokens:
        terminalreporter.write_line(
            f"  Cache-read:    {_session_metrics.llm_cache_read_tokens:,}"
        )
    terminalreporter.write_line(f"  Embed calls:   {_session_metrics.embed_calls}")
