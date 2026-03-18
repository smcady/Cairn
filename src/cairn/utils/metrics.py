"""Session-scoped LLM and embedding usage metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMetrics:
    """Accumulates API usage across a session (engine lifetime or eval scenario)."""

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0
    embed_calls: int = 0  # mirrors VectorIndex._request_count

    def record_llm(self, usage: Any) -> None:
        """Accumulate token counts from an Anthropic response.usage object."""
        self.llm_calls += 1
        self.llm_input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.llm_output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.llm_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.llm_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    @property
    def llm_total_tokens(self) -> int:
        return self.llm_input_tokens + self.llm_output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "llm_cache_read_tokens": self.llm_cache_read_tokens,
            "llm_cache_write_tokens": self.llm_cache_write_tokens,
            "embed_calls": self.embed_calls,
        }

    def format_summary(self) -> str:
        """One-line human-readable summary."""
        cache_note = ""
        if self.llm_cache_read_tokens:
            cache_note = f", {self.llm_cache_read_tokens:,} cache-read"
        return (
            f"{self.llm_calls} LLM call(s), "
            f"{self.llm_input_tokens:,} in / {self.llm_output_tokens:,} out"
            f"{cache_note}; "
            f"{self.embed_calls} embed call(s)"
        )
