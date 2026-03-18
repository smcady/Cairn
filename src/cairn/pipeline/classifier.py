"""Stage 1: structured-output LLM call to identify cognitive events in an exchange.

Returns ClassifiedEvent objects with node references expressed as plain-text descriptions.
The resolver (Stage 2) maps those descriptions to actual graph node IDs via vector search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from cairn.models.events import ClassifiedEvent, EventType
from cairn.models.graph_types import IdeaGraph, NodeStatus
from cairn.utils.metrics import SessionMetrics

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classifier.txt"


class ClassifierOutput(BaseModel):
    """Structured output from the classifier LLM call."""
    events: list[ClassifiedEvent] = Field(
        description="List of cognitive events identified in the exchange. "
        "Each event must have the correct event_type and all required fields for that type."
    )


class ClassifiedResult:
    """Resolved event: event_type + validated payload dict with real node IDs."""

    def __init__(self, event_type: EventType, payload: dict[str, Any], reasoning: str = ""):
        self.event_type = event_type
        self.payload = payload
        self.reasoning = reasoning


async def classify_exchange(
    client: AsyncAnthropic,
    exchange_text: str,
    graph: IdeaGraph,
    source: str = "user",
    model: str = "claude-sonnet-4-5-20250929",
    metrics: SessionMetrics | None = None,
) -> list[ClassifiedEvent]:
    """Classify an exchange into typed cognitive events (Stage 1).

    Returns ClassifiedEvent objects with node references as text descriptions.
    Call resolver.resolve_classified_event() on each result to get node IDs (Stage 2).

    Args:
        client: Anthropic async client
        exchange_text: The text of the exchange to classify
        graph: Current idea graph (active nodes only, no IDs passed to LLM)
        source: Who produced this exchange
        model: Model to use for classification
    """
    system_prompt = PROMPT_PATH.read_text()

    # Pass active nodes only, without IDs — classifier describes targets in text,
    # resolver handles ID lookup via vector search
    active_nodes = [
        {"type": n.type.value, "text": n.text}
        for n in graph.get_all_nodes()
        if n.status == NodeStatus.ACTIVE
    ]
    node_context = json.dumps(active_nodes, indent=2) if active_nodes else "No existing nodes yet."

    user_message = f"""EXISTING NODES (active only):
{node_context}

SOURCE: {source}

EXCHANGE:
{exchange_text}

Identify all cognitive events in this exchange. Each event must use the correct event_type and include all required fields for that type."""

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "name": "classify_events",
            "description": "Return all cognitive events identified in the exchange.",
            "input_schema": ClassifierOutput.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "classify_events"},
    )

    if metrics is not None:
        metrics.record_llm(response.usage)

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        return []

    result = ClassifierOutput.model_validate(tool_block.input)

    # Inject source into each event
    for event in result.events:
        if event.source == "user":
            event.source = source

    return result.events
