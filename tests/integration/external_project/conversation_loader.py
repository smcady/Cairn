"""Load test conversations from YAML fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Turn:
    user: str
    assistant: str

    @property
    def exchange(self) -> str:
        """Format as a single exchange string for MCP ingest."""
        return f"User: {self.user}\n\nAssistant: {self.assistant}"


@dataclass
class SearchAssertion:
    query: str
    min_results: int = 1


@dataclass
class OrientAssertion:
    topic: str
    must_not_be_empty: bool = True


@dataclass
class Assertions:
    min_nodes: int = 0
    has_superseded: bool = False
    search_queries: list[SearchAssertion] = field(default_factory=list)
    orient_topics: list[OrientAssertion] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    description: str
    domain: str
    turns: list[Turn]
    assertions: Assertions


def load_conversation(path: Path) -> Conversation:
    """Load a single conversation from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    turns = [Turn(user=t["user"].strip(), assistant=t["assistant"].strip()) for t in data["turns"]]

    assertions = Assertions()
    if "assertions" in data:
        a = data["assertions"]
        assertions.min_nodes = a.get("min_nodes", 0)
        assertions.has_superseded = a.get("has_superseded", False)
        assertions.search_queries = [
            SearchAssertion(**sq) for sq in a.get("search_queries", [])
        ]
        assertions.orient_topics = [
            OrientAssertion(**ot) for ot in a.get("orient_topics", [])
        ]

    return Conversation(
        id=data["id"],
        description=data.get("description", "").strip(),
        domain=data.get("domain", ""),
        turns=turns,
        assertions=assertions,
    )


def load_all_conversations(directory: Path | None = None) -> list[Conversation]:
    """Load all conversation YAML files from a directory."""
    if directory is None:
        directory = Path(__file__).parent / "conversations"
    files = sorted(directory.glob("*.yaml"))
    return [load_conversation(f) for f in files]
