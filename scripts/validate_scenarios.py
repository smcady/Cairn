"""Validate conversation scenarios against the current classifier.

Ingests each scenario turn-by-turn through the full pipeline and reports:
- Event types produced per turn
- SUPPORT/CONTRADICTION rates
- Evidence text quality (for assessing whether strength classification would work)
- Applied vs dropped event counts
- Confidence distribution after all turns

Usage:
    .venv/bin/python scripts/validate_scenarios.py
    .venv/bin/python scripts/validate_scenarios.py --scenario saas_churn_analysis
    .venv/bin/python scripts/validate_scenarios.py --scenarios-dir tests/integration/external_project/conversations
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

# Load .env.local before any src imports
_env_file = Path(__file__).parent.parent / ".env.local"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=True)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "tests" / "integration" / "external_project"))

from conversation_loader import load_all_conversations, load_conversation

from cairn.memory.engine import MemoryEngine
from cairn.models.events import EventLog, EventType
from cairn.models.graph_types import IdeaGraph, NodeStatus
from cairn.utils.vector_index import VectorIndex


def make_engine() -> MemoryEngine:
    """Fresh in-memory engine for each scenario."""
    return MemoryEngine(
        event_log=EventLog(":memory:"),
        graph=IdeaGraph(),
        vector_index=VectorIndex(":memory:", max_requests=0),
    )


def _bar(count: int, max_width: int = 30) -> str:
    return "#" * min(count, max_width)


async def validate_scenario(conv, verbose: bool = False) -> dict:
    """Run a single conversation through the pipeline and collect stats."""
    engine = make_engine()
    print(f"\n{'=' * 60}")
    print(f"  {conv.id} ({conv.domain})")
    print(f"  {conv.description[:80]}")
    print(f"{'=' * 60}")

    all_events = []
    all_dropped = []
    per_turn = []

    for i, turn in enumerate(conv.turns, 1):
        result = await engine.ingest(turn.exchange, source=f"{conv.id}-turn-{i}")

        turn_events = [(e.event_type.value, e.payload, e.payload.get("domain", "general")) for e in result.applied_events]
        turn_dropped = [(d.event_type, d.reason) for d in result.dropped_events]

        per_turn.append({
            "turn": i,
            "applied": turn_events,
            "dropped": turn_dropped,
        })

        all_events.extend(result.applied_events)
        all_dropped.extend(result.dropped_events)

        # Per-turn summary
        print(f"\n  Turn {i}: {len(turn_events)} applied, {len(turn_dropped)} dropped")
        for et, _, domain in turn_events:
            domain_tag = f" [{domain}]" if domain != "general" else ""
            print(f"    + {et}{domain_tag}")
        for et, reason in turn_dropped:
            print(f"    - {et} ({reason})")

        # Show evidence/objection text for SUPPORT/CONTRADICTION
        for et, payload, _ in turn_events:
            if et == "SUPPORT" and verbose:
                print(f"      evidence: {payload.get('evidence_text', '')[:100]}")
            elif et == "CONTRADICTION" and verbose:
                print(f"      objection: {payload.get('objection_text', '')[:100]}")

    # Aggregate stats
    total_applied = len(all_events)
    total_dropped = len(all_dropped)
    total = total_applied + total_dropped
    drop_rate = total_dropped / total if total > 0 else 0

    event_counts = Counter(e.event_type.value for e in all_events)
    domain_counts = Counter(e.payload.get("domain", "general") for e in all_events)

    print(f"\n  --- Summary ---")
    print(f"  Total events: {total} ({total_applied} applied, {total_dropped} dropped)")
    print(f"  Drop rate: {drop_rate:.1%}")
    print(f"\n  Event type distribution:")
    for et, count in event_counts.most_common():
        print(f"    {et:25s}: {count:3d} {_bar(count)}")

    print(f"\n  Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"    {domain:25s}: {count:3d} {_bar(count)}")

    # Confidence distribution
    nodes = engine.graph.get_all_nodes()
    active = [n for n in nodes if n.status == NodeStatus.ACTIVE]
    conf_dist = Counter(round(n.confidence, 1) for n in active)

    print(f"\n  Confidence distribution ({len(active)} active nodes):")
    for c in sorted(conf_dist.keys()):
        print(f"    {c:.1f}: {conf_dist[c]:3d} {_bar(conf_dist[c])}")

    # SUPPORT/CONTRADICTION analysis
    support_count = event_counts.get("SUPPORT", 0)
    contra_count = event_counts.get("CONTRADICTION", 0)
    prop_count = event_counts.get("NEW_PROPOSITION", 0)

    print(f"\n  Key ratios:")
    print(f"    SUPPORT / NEW_PROPOSITION:       {support_count}/{prop_count}")
    print(f"    CONTRADICTION / NEW_PROPOSITION:  {contra_count}/{prop_count}")

    # Evidence text analysis (for SUPPORT/CONTRADICTION events)
    evidence_texts = []
    for e in all_events:
        if e.event_type == EventType.SUPPORT:
            evidence_texts.append(("SUPPORT", e.payload.get("evidence_text", ""), e.payload.get("evidence_strength", 0.5)))
        elif e.event_type == EventType.CONTRADICTION:
            evidence_texts.append(("CONTRADICTION", e.payload.get("objection_text", ""), e.payload.get("evidence_strength", 0.5)))

    if evidence_texts:
        print(f"\n  Evidence/objection texts ({len(evidence_texts)} total):")
        for et, text, strength in evidence_texts:
            word_count = len(text.split())
            has_numbers = any(c.isdigit() for c in text)
            qualifier = "quantitative" if has_numbers else "qualitative"
            print(f"    [{et}] strength={strength:.2f} ({word_count} words, {qualifier})")
            print(f"      {text[:120]}")

    return {
        "id": conv.id,
        "domain": conv.domain,
        "total_applied": total_applied,
        "total_dropped": total_dropped,
        "drop_rate": drop_rate,
        "event_counts": dict(event_counts),
        "support_count": support_count,
        "contradiction_count": contra_count,
        "proposition_count": prop_count,
        "active_nodes": len(active),
        "confidence_distribution": {str(k): v for k, v in conf_dist.items()},
    }


async def main(args: argparse.Namespace) -> None:
    scenarios_dir = Path(args.scenarios_dir)

    if args.scenario:
        # Load a specific scenario
        files = sorted(scenarios_dir.glob("*.yaml"))
        matching = [f for f in files if args.scenario in f.stem]
        if not matching:
            print(f"No scenario matching '{args.scenario}' found in {scenarios_dir}")
            sys.exit(1)
        conversations = [load_conversation(f) for f in matching]
    else:
        conversations = load_all_conversations(scenarios_dir)

    print(f"Loaded {len(conversations)} scenarios from {scenarios_dir}")

    results = []
    for conv in conversations:
        result = await validate_scenario(conv, verbose=args.verbose)
        results.append(result)

    # Cross-scenario comparison
    print(f"\n\n{'=' * 60}")
    print("  Cross-Scenario Comparison")
    print(f"{'=' * 60}")
    print(f"\n  {'Scenario':<30} {'Domain':<20} {'Props':>5} {'Supp':>5} {'Contr':>5} {'Drop%':>6}")
    print(f"  {'-'*30} {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    for r in results:
        print(
            f"  {r['id']:<30} {r['domain']:<20} "
            f"{r['proposition_count']:>5} {r['support_count']:>5} "
            f"{r['contradiction_count']:>5} {r['drop_rate']:>5.0%}"
        )

    # Overall assessment
    total_support = sum(r["support_count"] for r in results)
    total_contra = sum(r["contradiction_count"] for r in results)
    total_props = sum(r["proposition_count"] for r in results)

    print(f"\n  Overall: {total_support} SUPPORT, {total_contra} CONTRADICTION, {total_props} NEW_PROPOSITION")
    if total_props > 0:
        print(f"  SUPPORT rate: {total_support/total_props:.1%} of propositions")
        print(f"  CONTRADICTION rate: {total_contra/total_props:.1%} of propositions")

    # Decision gate assessment
    print(f"\n  --- Decision Gate ---")
    if total_support >= len(conversations) and total_contra >= 1:
        print("  PASS: Classifier produces SUPPORT/CONTRADICTION at reasonable rates.")
        print("  Proceed to Phase 1 (evidence strength).")
    elif total_support > 0 or total_contra > 0:
        print("  MARGINAL: Some SUPPORT/CONTRADICTION events but below expectations.")
        print("  Consider tuning the classifier prompt before Phase 1.")
    else:
        print("  FAIL: No SUPPORT/CONTRADICTION events produced.")
        print("  Classifier prompt needs significant tuning before Phase 1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate conversation scenarios against the current classifier."
    )
    parser.add_argument(
        "--scenarios-dir",
        default="tests/integration/external_project/conversations",
        help="Directory containing conversation YAML files",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only scenarios matching this string",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show evidence/objection text inline per turn",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
