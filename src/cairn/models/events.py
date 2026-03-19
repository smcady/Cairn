"""Event type hierarchy and EventLog with SQLite persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    NEW_PROPOSITION = "NEW_PROPOSITION"
    SUPPORT = "SUPPORT"
    CONTRADICTION = "CONTRADICTION"
    REFINEMENT = "REFINEMENT"
    NEW_QUESTION = "NEW_QUESTION"
    QUESTION_RESOLVED = "QUESTION_RESOLVED"
    CONNECTION = "CONNECTION"
    TENSION_IDENTIFIED = "TENSION_IDENTIFIED"
    TERRITORY_IDENTIFIED = "TERRITORY_IDENTIFIED"
    REFRAME = "REFRAME"
    SYNTHESIS = "SYNTHESIS"
    ABANDONMENT = "ABANDONMENT"


# --- Event Payloads ---

class NewPropositionPayload(BaseModel):
    text: str
    source: str = "user"
    related_node_ids: list[str] = Field(default_factory=list)
    node_id: str = ""  # pre-assigned by engine; empty means mutator generates


class SupportPayload(BaseModel):
    target_node_id: str
    evidence_text: str
    source: str = "user"
    evidence_node_id: str = ""  # pre-assigned by engine
    evidence_strength: float = 0.5  # 0.0-1.0 scale from classifier


class ContradictionPayload(BaseModel):
    target_node_id: str
    objection_text: str
    source: str = "user"
    objection_node_id: str = ""  # pre-assigned by engine
    evidence_strength: float = 0.5  # 0.0-1.0 scale from classifier


class RefinementPayload(BaseModel):
    target_node_id: str
    new_text: str
    source: str = "user"


class NewQuestionPayload(BaseModel):
    text: str
    related_node_ids: list[str] = Field(default_factory=list)
    source: str = "user"
    node_id: str = ""  # pre-assigned by engine


class QuestionResolvedPayload(BaseModel):
    question_node_id: str
    resolution_text: str
    source: str = "user"
    resolution_node_id: str = ""  # pre-assigned by engine


class ConnectionPayload(BaseModel):
    source_node_id: str
    target_node_id: str
    basis: str = ""


class TensionIdentifiedPayload(BaseModel):
    node_ids: list[str] = Field(min_length=2)
    description: str
    source: str = "user"
    node_id: str = ""  # pre-assigned by engine


class TerritoryIdentifiedPayload(BaseModel):
    text: str
    adjacent_node_ids: list[str] = Field(default_factory=list)
    source: str = "user"
    node_id: str = ""  # pre-assigned by engine


class ReframePayload(BaseModel):
    text: str
    affected_node_ids: list[str] = Field(default_factory=list)
    source: str = "user"
    node_id: str = ""  # pre-assigned by engine


class SynthesisPayload(BaseModel):
    text: str
    constituent_node_ids: list[str] = Field(min_length=2)
    source: str = "user"
    supersedes_constituents: bool = True
    node_id: str = ""  # pre-assigned by engine


class AbandonmentPayload(BaseModel):
    target_node_id: str
    reason: str = ""


PAYLOAD_MAP: dict[EventType, type[BaseModel]] = {
    EventType.NEW_PROPOSITION: NewPropositionPayload,
    EventType.SUPPORT: SupportPayload,
    EventType.CONTRADICTION: ContradictionPayload,
    EventType.REFINEMENT: RefinementPayload,
    EventType.NEW_QUESTION: NewQuestionPayload,
    EventType.QUESTION_RESOLVED: QuestionResolvedPayload,
    EventType.CONNECTION: ConnectionPayload,
    EventType.TENSION_IDENTIFIED: TensionIdentifiedPayload,
    EventType.TERRITORY_IDENTIFIED: TerritoryIdentifiedPayload,
    EventType.REFRAME: ReframePayload,
    EventType.SYNTHESIS: SynthesisPayload,
    EventType.ABANDONMENT: AbandonmentPayload,
}


class Event(BaseModel):
    id: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: EventType
    payload: dict[str, Any]
    turn_number: int | None = None
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    workspace_id: str = ""
    parent_hash: str = ""

    def get_typed_payload(self) -> BaseModel:
        payload_cls = PAYLOAD_MAP[self.event_type]
        return payload_cls.model_validate(self.payload)


def _compute_event_hash(event: Event) -> str:
    """SHA-256 of an event's canonical content (all fields including parent_hash)."""
    content = json.dumps({
        "id": event.id,
        "timestamp": event.timestamp,
        "event_type": event.event_type.value,
        "payload": event.payload,
        "turn_number": event.turn_number,
        "session_id": event.session_id,
        "workspace_id": event.workspace_id,
        "parent_hash": event.parent_hash,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def validate_event_payload(event_type: EventType, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate and normalize a payload dict against its expected schema.

    Returns (validated_payload_dict, "") on success,
    or (None, error_description) on failure.
    """
    payload_cls = PAYLOAD_MAP[event_type]

    try:
        validated = payload_cls.model_validate(payload)
        return validated.model_dump(), ""
    except Exception as e:
        return None, f"{payload_cls.__name__}: {e}"


# --- Flat Classified Event model (for structured LLM output) ---
# Single model with all fields optional except event_type.
# Descriptions tell the LLM which fields to fill per event type.
# This avoids the grammar-too-large error from a 12-variant discriminated union.

class ClassifiedEvent(BaseModel):
    """A single classified cognitive event. Fill the fields relevant to the event_type.

    Node references are expressed as plain-text descriptions, not IDs.
    The resolver stage maps descriptions to actual graph node IDs via vector search.
    """

    event_type: EventType = Field(description="The type of cognitive event identified")
    reasoning: str = Field(default="", description="Brief reasoning for why this event was identified")

    # Text content — required for: NEW_PROPOSITION, NEW_QUESTION, SYNTHESIS, TERRITORY_IDENTIFIED, REFRAME
    text: str = Field(default="", description=(
        "Main text content. Required for NEW_PROPOSITION, NEW_QUESTION, "
        "SYNTHESIS, TERRITORY_IDENTIFIED, REFRAME."
    ))

    source: str = Field(default="user", description="Who introduced this: 'user' or 'system'")

    # Target node description — required for: SUPPORT, CONTRADICTION, REFINEMENT, ABANDONMENT, CONNECTION
    target_node_description: str = Field(default="", description=(
        "Plain-text description of the target node (enough to identify it unambiguously). "
        "Required for SUPPORT, CONTRADICTION, REFINEMENT, ABANDONMENT. "
        "Also the target in CONNECTION."
    ))

    # Related nodes — used by: NEW_PROPOSITION, NEW_QUESTION
    related_node_descriptions: list[str] = Field(default_factory=list, description=(
        "Plain-text descriptions of related existing nodes. Used by NEW_PROPOSITION, NEW_QUESTION."
    ))

    # SUPPORT
    evidence_text: str = Field(default="", description="Evidence text. Required for SUPPORT.")

    # CONTRADICTION
    objection_text: str = Field(default="", description="Objection or challenge text. Required for CONTRADICTION.")

    # REFINEMENT
    new_text: str = Field(default="", description="Refined version of text. Required for REFINEMENT.")

    # QUESTION_RESOLVED
    question_node_description: str = Field(default="", description="Description of the question being resolved. Required for QUESTION_RESOLVED.")
    resolution_text: str = Field(default="", description="Resolution or answer. Required for QUESTION_RESOLVED.")

    # CONNECTION
    source_node_description: str = Field(default="", description="Description of source node. Required for CONNECTION.")
    basis: str = Field(default="", description="Basis for a CONNECTION between nodes.")

    # TENSION_IDENTIFIED
    node_descriptions: list[str] = Field(default_factory=list, description="Descriptions of nodes involved in tension. Required for TENSION_IDENTIFIED (at least 2).")
    description: str = Field(default="", description="Description of the tension. Required for TENSION_IDENTIFIED.")

    # TERRITORY_IDENTIFIED
    adjacent_node_descriptions: list[str] = Field(default_factory=list, description="Descriptions of adjacent nodes. Used by TERRITORY_IDENTIFIED.")

    # REFRAME
    affected_node_descriptions: list[str] = Field(default_factory=list, description="Descriptions of affected nodes. Used by REFRAME.")

    # SYNTHESIS
    constituent_node_descriptions: list[str] = Field(default_factory=list, description="Descriptions of constituent nodes. Required for SYNTHESIS (at least 2).")
    supersedes_constituents: bool = Field(default=True, description="If true, constituent nodes are marked SUPERSEDED. Set false for additive synthesis where constituents remain independently valid.")

    # ABANDONMENT
    reason: str = Field(default="", description="Reason for abandonment. Used by ABANDONMENT.")

    # Evidence strength (SUPPORT and CONTRADICTION only)
    evidence_strength: float = Field(default=0.5, ge=0.0, le=1.0, description=(
        "Strength of evidence for SUPPORT or CONTRADICTION events. "
        "Scale: 0.0-0.2 (anecdote/opinion), 0.2-0.4 (authority claim), "
        "0.4-0.6 (logical argument), 0.6-0.8 (empirical/quantitative data), "
        "0.8-1.0 (proven/replicated result). Default 0.5 for non-evidence events."
    ))


# --- EventLog with SQLite persistence ---

class EventLog:
    """Append-only event log backed by SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                turn_number INTEGER,
                session_id TEXT,
                workspace_id TEXT NOT NULL DEFAULT '',
                parent_hash TEXT NOT NULL DEFAULT ''
            )
        """)
        # Migrate existing tables that predate hash-linking
        try:
            self._conn.execute("ALTER TABLE events ADD COLUMN parent_hash TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self._conn.commit()

    def _get_last_event_hash(self) -> str:
        """Return the hash of the most recently appended event, or '' for an empty log."""
        row = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return ""
        return _compute_event_hash(self._row_to_event(row))

    def append(self, event: Event) -> Event:
        event.parent_hash = self._get_last_event_hash()
        cursor = self._conn.execute(
            """INSERT INTO events (timestamp, event_type, payload, turn_number, session_id, workspace_id, parent_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.timestamp,
                event.event_type.value,
                json.dumps(event.payload),
                event.turn_number,
                event.session_id,
                event.workspace_id,
                event.parent_hash,
            ),
        )
        self._conn.commit()
        event.id = cursor.lastrowid
        return event

    def get_all(self) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_since(self, event_id: int) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE id > ? ORDER BY id ASC", (event_id,)
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_by_workspace(self, workspace_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE workspace_id = ? ORDER BY id ASC",
            (workspace_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_by_session(self, session_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_recent(self, n: int = 10) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            timestamp=row["timestamp"],
            event_type=EventType(row["event_type"]),
            payload=json.loads(row["payload"]),
            turn_number=row["turn_number"],
            session_id=row["session_id"] or "",
            workspace_id=row["workspace_id"] or "",
            parent_hash=row["parent_hash"] if "parent_hash" in row.keys() else "",
        )

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain is intact. Returns (ok, message)."""
        events = self.get_all()
        prev_hash = ""
        for event in events:
            if event.parent_hash != prev_hash:
                return False, f"Chain broken at event id={event.id}: expected parent_hash={prev_hash!r}, got {event.parent_hash!r}"
            prev_hash = _compute_event_hash(event)
        return True, f"Chain intact ({len(events)} events)"

    def close(self) -> None:
        self._conn.close()
