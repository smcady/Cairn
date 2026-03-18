"""Cairn eval harness — scores pipeline quality against scenario-defined ground truth.

Usage:
    python scripts/run_evals.py                          # full eval with API keys
    python scripts/run_evals.py --skip-recall            # structural checks only, no API
    python scripts/run_evals.py --scenario arch_debate_01
    python scripts/run_evals.py --threshold 0.70 --json-out evals/report.json
    python scripts/run_evals.py --verbose

Exit codes:
    0  overall score >= threshold (or structural-only mode with adjusted threshold)
    1  below threshold
    2  setup error (missing scenario files, schema error, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# Load .env.local before any src imports.
_env_file = Path(__file__).parent.parent / ".env.local"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# Resolve project root so src/ imports work regardless of cwd.
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from cairn.memory.engine import MemoryEngine
from cairn.models.events import EventLog, EventType
from cairn.models.graph_types import IdeaGraph
from cairn.pipeline.renderer import render_structured_summary
from cairn.utils.metrics import SessionMetrics
from cairn.utils.vector_index import VectorIndex


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalScenario:
    scenario_id: str
    description: str
    domain: str
    difficulty: str
    turns: list[dict]
    expected_event_types: list[dict]
    recall_queries: list[dict]
    render_must_mention: list[str]
    quality_gates: dict


@dataclass
class EventTypeResult:
    event_type: str
    expected_at_least: int
    actual_count: int
    passed: bool


@dataclass
class RecallQueryResult:
    query: str
    top_k: int
    hit: bool
    best_score: float | None
    matched_keyword: str | None
    top_result_text: str | None


@dataclass
class RenderResult:
    keyword: str
    found: bool


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    domain: str
    difficulty: str
    # Ingest stats
    total_turns: int
    total_events_seen: int
    applied_events: int
    dropped_events: int
    drop_rate: float
    final_node_count: int
    final_edge_count: int
    # Scores
    event_type_results: list[EventTypeResult]
    event_type_recall: float
    recall_results: list[RecallQueryResult]
    recall_hit_rate: float
    recall_skipped: bool
    render_results: list[RenderResult]
    render_coverage: float
    overall_score: float
    # Gates
    gate_drop_rate_ok: bool
    gate_min_nodes_ok: bool
    passed: bool
    failure_reasons: list[str]
    # API usage
    metrics: dict = field(default_factory=dict)


@dataclass
class AggregateReport:
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    mean_event_type_recall: float
    mean_recall_hit_rate: float
    mean_render_coverage: float
    mean_drop_rate: float
    overall_score: float
    threshold: float
    passed_threshold: bool
    recall_skipped: bool
    recall_skip_reason: str  # "flag" | "no_keys" | ""
    per_scenario: list[ScenarioResult]
    # Aggregate API usage
    total_llm_calls: int = 0
    total_llm_input_tokens: int = 0
    total_llm_output_tokens: int = 0
    total_llm_cache_read_tokens: int = 0
    total_embed_calls: int = 0


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"scenario_id", "description", "domain", "difficulty", "turns", "expectations"}
_DIFFICULTIES = {"easy", "medium", "hard"}
_DOMAINS = {"technical_architecture", "product_strategy", "ideation", "mixed"}


def load_scenario(path: Path) -> EvalScenario:
    with open(path) as f:
        data = yaml.safe_load(f)

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"{path.name}: missing fields: {missing}")

    if data["difficulty"] not in _DIFFICULTIES:
        raise ValueError(f"{path.name}: difficulty must be one of {_DIFFICULTIES}")

    exp = data.get("expectations", {})
    return EvalScenario(
        scenario_id=data["scenario_id"],
        description=data["description"],
        domain=data.get("domain", "unknown"),
        difficulty=data["difficulty"],
        turns=data["turns"],
        expected_event_types=exp.get("event_types", []),
        recall_queries=exp.get("recall_queries", []),
        render_must_mention=exp.get("render_must_mention", []),
        quality_gates=exp.get("quality_gates", {}),
    )


def load_all_scenarios(scenarios_dir: Path, scenario_id: str | None = None) -> list[EvalScenario]:
    paths = sorted(scenarios_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No scenario YAML files found in {scenarios_dir}")
    scenarios = [load_scenario(p) for p in paths]
    if scenario_id:
        scenarios = [s for s in scenarios if s.scenario_id == scenario_id]
        if not scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found in {scenarios_dir}")
    return scenarios


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine() -> MemoryEngine:
    """Fresh in-memory engine — max_requests=0 disables billing cap for intentional eval use."""
    return MemoryEngine(
        event_log=EventLog(":memory:"),
        graph=IdeaGraph(),
        vector_index=VectorIndex(":memory:", max_requests=0),
    )


# ---------------------------------------------------------------------------
# Evaluation checks
# ---------------------------------------------------------------------------

def check_event_types(
    applied: list,
    expectations: list[dict],
) -> tuple[list[EventTypeResult], float]:
    """Score event type recall. Returns (results, recall_score)."""
    # Count actual event types
    actual_counts: dict[str, int] = {}
    for event in applied:
        et = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        actual_counts[et] = actual_counts.get(et, 0) + 1

    results = []
    tp = 0
    fn = 0
    for exp in expectations:
        et_name = exp["type"]
        at_least = exp.get("at_least", 1)
        actual = actual_counts.get(et_name, 0)
        passed = actual >= at_least
        if passed:
            tp += 1
        else:
            fn += 1
        results.append(EventTypeResult(
            event_type=et_name,
            expected_at_least=at_least,
            actual_count=actual,
            passed=passed,
        ))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # no expectations = perfect
    return results, recall


async def check_recall(
    engine: MemoryEngine,
    queries: list[dict],
) -> tuple[list[RecallQueryResult], float]:
    """Score semantic recall. Returns (results, hit_rate)."""
    results = []
    hits = 0

    for q in queries:
        query_text = q["query"]
        keywords = [kw.lower() for kw in q.get("must_contain_keywords", [])]
        top_k = q.get("top_k", 3)

        search_results = await engine.search_nodes(query_text, k=top_k)

        hit = False
        best_score = None
        matched_kw = None
        top_text = None

        if search_results:
            best_score = search_results[0][1]
            top_text = search_results[0][0].text

        for node, score in search_results:
            node_text_lower = node.text.lower()
            for kw in keywords:
                if kw in node_text_lower:
                    hit = True
                    matched_kw = kw
                    break
            if hit:
                break

        if hit:
            hits += 1

        results.append(RecallQueryResult(
            query=query_text,
            top_k=top_k,
            hit=hit,
            best_score=round(best_score, 4) if best_score is not None else None,
            matched_keyword=matched_kw,
            top_result_text=top_text,
        ))

    hit_rate = hits / len(queries) if queries else 1.0
    return results, hit_rate


def check_render(
    engine: MemoryEngine,
    keywords: list[str],
) -> tuple[list[RenderResult], float]:
    """Score render coverage. Returns (results, coverage_score)."""
    if not keywords:
        return [], 1.0

    render_output = render_structured_summary(engine.graph).lower()
    results = []
    found_count = 0

    for kw in keywords:
        found = kw.lower() in render_output
        if found:
            found_count += 1
        results.append(RenderResult(keyword=kw, found=found))

    coverage = found_count / len(keywords)
    return results, coverage


def compute_overall_score(
    event_type_recall: float,
    recall_hit_rate: float,
    render_coverage: float,
    drop_rate: float,
    recall_skipped: bool,
) -> float:
    """Weighted composite score. Rebalances weights when recall is skipped."""
    drop_score = max(0.0, 1.0 - drop_rate)

    if recall_skipped:
        # Rebalance: ET=0.57, render=0.22, drop=0.14, recall=0.00 → sum=0.93 → renorm
        score = (
            0.57 * event_type_recall +
            0.22 * render_coverage +
            0.14 * drop_score
        ) / 0.93
    else:
        score = (
            0.40 * event_type_recall +
            0.35 * recall_hit_rate +
            0.15 * render_coverage +
            0.10 * drop_score
        )

    return round(score, 4)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

async def run_scenario(
    scenario: EvalScenario,
    skip_recall: bool = False,
    verbose: bool = False,
) -> ScenarioResult:
    engine = make_engine()

    all_applied = []
    all_dropped = []

    for turn in scenario.turns:
        text = turn["text"] if isinstance(turn, dict) else turn
        source = turn.get("source", "user") if isinstance(turn, dict) else "user"
        result = await engine.ingest(text, source=source)
        all_applied.extend(result.applied_events)
        all_dropped.extend(result.dropped_events)

    total_events = len(all_applied) + len(all_dropped)
    drop_rate = len(all_dropped) / total_events if total_events > 0 else 0.0

    # Event type check
    et_results, et_recall = check_event_types(all_applied, scenario.expected_event_types)

    # Recall check
    if skip_recall or not engine.vector_index or len(engine.vector_index) == 0:
        recall_results = []
        recall_hit_rate = 0.0
        recall_skipped = True
    else:
        recall_results, recall_hit_rate = await check_recall(engine, scenario.recall_queries)
        recall_skipped = False

    # Render check
    render_results, render_coverage = check_render(engine, scenario.render_must_mention)

    # Gates
    max_drop = scenario.quality_gates.get("max_drop_rate", 0.40)
    min_nodes = scenario.quality_gates.get("min_nodes", 2)
    node_count = engine.graph.node_count()
    edge_count = engine.graph.edge_count()

    gate_drop_ok = drop_rate <= max_drop
    gate_nodes_ok = node_count >= min_nodes

    # Score
    overall = compute_overall_score(
        et_recall, recall_hit_rate, render_coverage, drop_rate, recall_skipped
    )

    # Failures
    failure_reasons = []
    for r in et_results:
        if not r.passed:
            failure_reasons.append(
                f"event type {r.event_type}: expected ≥{r.expected_at_least}, got {r.actual_count}"
            )
    if not gate_drop_ok:
        failure_reasons.append(f"drop_rate {drop_rate:.2f} > max {max_drop}")
    if not gate_nodes_ok:
        failure_reasons.append(f"node_count {node_count} < min {min_nodes}")
    for r in recall_results:
        if not r.hit:
            failure_reasons.append(f"recall miss: '{r.query[:50]}...' keywords={r.matched_keyword}")

    passed = (
        len([r for r in et_results if not r.passed]) == 0
        and gate_drop_ok
        and gate_nodes_ok
    )

    scenario_metrics = engine.metrics.to_dict()
    # embed_calls already synced into metrics by get_stats(); call it to trigger sync
    engine.get_stats()
    scenario_metrics = engine.metrics.to_dict()

    if verbose:
        _print_verbose_scenario(scenario, all_applied, all_dropped, et_results, recall_results, render_results, metrics=scenario_metrics)

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        domain=scenario.domain,
        difficulty=scenario.difficulty,
        total_turns=len(scenario.turns),
        total_events_seen=total_events,
        applied_events=len(all_applied),
        dropped_events=len(all_dropped),
        drop_rate=round(drop_rate, 4),
        final_node_count=node_count,
        final_edge_count=edge_count,
        event_type_results=et_results,
        event_type_recall=round(et_recall, 4),
        recall_results=recall_results,
        recall_hit_rate=round(recall_hit_rate, 4),
        recall_skipped=recall_skipped,
        render_results=render_results,
        render_coverage=round(render_coverage, 4),
        overall_score=overall,
        gate_drop_rate_ok=gate_drop_ok,
        gate_min_nodes_ok=gate_nodes_ok,
        passed=passed,
        failure_reasons=failure_reasons,
        metrics=scenario_metrics,
    )


# ---------------------------------------------------------------------------
# Aggregate + report
# ---------------------------------------------------------------------------

def aggregate(results: list[ScenarioResult], threshold: float, recall_skip_reason: str = "") -> AggregateReport:
    n = len(results)
    if n == 0:
        raise ValueError("No scenario results to aggregate")

    recall_skipped = any(r.recall_skipped for r in results)

    mean_et = sum(r.event_type_recall for r in results) / n
    mean_recall = sum(r.recall_hit_rate for r in results) / n
    mean_render = sum(r.render_coverage for r in results) / n
    mean_drop = sum(r.drop_rate for r in results) / n

    overall = compute_overall_score(
        mean_et, mean_recall, mean_render, mean_drop, recall_skipped
    )

    return AggregateReport(
        total_scenarios=n,
        passed_scenarios=sum(1 for r in results if r.passed),
        failed_scenarios=sum(1 for r in results if not r.passed),
        mean_event_type_recall=round(mean_et, 4),
        mean_recall_hit_rate=round(mean_recall, 4),
        mean_render_coverage=round(mean_render, 4),
        mean_drop_rate=round(mean_drop, 4),
        overall_score=round(overall, 4),
        threshold=threshold,
        passed_threshold=overall >= threshold,
        recall_skipped=recall_skipped,
        recall_skip_reason=recall_skip_reason,
        per_scenario=results,
        total_llm_calls=sum(r.metrics.get("llm_calls", 0) for r in results),
        total_llm_input_tokens=sum(r.metrics.get("llm_input_tokens", 0) for r in results),
        total_llm_output_tokens=sum(r.metrics.get("llm_output_tokens", 0) for r in results),
        total_llm_cache_read_tokens=sum(r.metrics.get("llm_cache_read_tokens", 0) for r in results),
        total_embed_calls=sum(r.metrics.get("embed_calls", 0) for r in results),
    )


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_report(report: AggregateReport) -> str:
    lines = []
    w = 58
    lines.append("=" * w)
    lines.append("  Cairn Eval Report")
    lines.append("=" * w)

    if report.recall_skip_reason == "flag":
        recall_note = " (recall skipped — --skip-recall flag)"
    elif report.recall_skip_reason == "no_keys":
        recall_note = " (recall skipped — API keys not set)"
    else:
        recall_note = ""
    threshold_status = _status(report.passed_threshold)
    lines.append(
        f"  Overall score: {report.overall_score:.3f} "
        f"{'≥' if report.passed_threshold else '<'} "
        f"{report.threshold:.2f}  {threshold_status}{recall_note}"
    )
    lines.append("-" * w)
    lines.append(
        f"  {'Scenario':<30} {'Diff':<8} {'Score':>6}  {'Status'}"
    )
    lines.append("-" * w)

    for s in report.per_scenario:
        status = _status(s.passed)
        lines.append(
            f"  {s.scenario_id:<30} {s.difficulty:<8} {s.overall_score:>6.3f}  {status}"
        )
        if not s.passed:
            for reason in s.failure_reasons[:2]:  # show first 2
                lines.append(f"    ↳ {reason}")

    lines.append("-" * w)
    lines.append(f"  Event type recall:  {report.mean_event_type_recall:.3f}")
    if not report.recall_skipped:
        lines.append(f"  Recall hit rate:    {report.mean_recall_hit_rate:.3f}")
    else:
        lines.append(f"  Recall hit rate:    (skipped)")
    lines.append(f"  Render coverage:    {report.mean_render_coverage:.3f}")
    lines.append(f"  Mean drop rate:     {report.mean_drop_rate:.3f}")
    lines.append(f"  Passed:  {report.passed_scenarios}/{report.total_scenarios}")
    lines.append("-" * w)
    lines.append("  API Usage")
    lines.append(f"  LLM calls:          {report.total_llm_calls}")
    lines.append(f"  Input tokens:       {report.total_llm_input_tokens:,}")
    lines.append(f"  Output tokens:      {report.total_llm_output_tokens:,}")
    total_tokens = report.total_llm_input_tokens + report.total_llm_output_tokens
    lines.append(f"  Total tokens:       {total_tokens:,}")
    if report.total_llm_cache_read_tokens:
        lines.append(f"  Cache-read tokens:  {report.total_llm_cache_read_tokens:,}")
    lines.append(f"  Embed calls:        {report.total_embed_calls}")
    lines.append("=" * w)

    if report.recall_skip_reason == "flag":
        lines.append("\n  [!] Recall queries skipped (--skip-recall). Structural score only.")
    elif report.recall_skip_reason == "no_keys":
        lines.append(
            "\n  [!] Recall queries skipped — set ANTHROPIC_API_KEY + VOYAGE_API_KEY\n"
            "      for full evaluation. Structural score only."
        )

    return "\n".join(lines)


def _print_verbose_scenario(
    scenario: EvalScenario,
    applied: list,
    dropped: list,
    et_results: list[EventTypeResult],
    recall_results: list[RecallQueryResult],
    render_results: list[RenderResult],
    metrics: dict | None = None,
) -> None:
    print(f"\n  [{scenario.scenario_id}] {scenario.description[:60]}")
    if metrics:
        total = metrics.get("llm_input_tokens", 0) + metrics.get("llm_output_tokens", 0)
        print(
            f"  API: {metrics.get('llm_calls', 0)} LLM call(s), "
            f"{metrics.get('llm_input_tokens', 0):,} in / "
            f"{metrics.get('llm_output_tokens', 0):,} out "
            f"({total:,} total); "
            f"{metrics.get('embed_calls', 0)} embed call(s)"
        )
    print(f"  Applied events ({len(applied)}):")
    for e in applied:
        et = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        print(f"    + {et}")
    if dropped:
        print(f"  Dropped events ({len(dropped)}):")
        for d in dropped:
            score = f" score={d.resolution_score:.3f}" if d.resolution_score else ""
            print(f"    - {d.event_type}: {d.reason}{score}")
    print("  Event type checks:")
    for r in et_results:
        sym = "✓" if r.passed else "✗"
        print(f"    {sym} {r.event_type}: expected ≥{r.expected_at_least}, got {r.actual_count}")
    if recall_results:
        print("  Recall queries:")
        for r in recall_results:
            sym = "✓" if r.hit else "✗"
            score_str = f" ({r.best_score:.3f})" if r.best_score else ""
            print(f"    {sym} '{r.query[:45]}...'{score_str}")
            if not r.hit and r.top_result_text:
                print(f"       top result: {r.top_result_text[:80]!r}")
    if render_results:
        print("  Render coverage:")
        for r in render_results:
            sym = "✓" if r.found else "✗"
            print(f"    {sym} '{r.keyword}'")


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

def _has_api_keys() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(os.environ.get("VOYAGE_API_KEY"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        print(f"Error: scenarios directory not found: {scenarios_dir}", file=sys.stderr)
        return 2

    try:
        scenarios = load_all_scenarios(scenarios_dir, scenario_id=args.scenario)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading scenarios: {e}", file=sys.stderr)
        return 2

    has_keys = _has_api_keys()
    skip_recall = args.skip_recall or not has_keys
    if args.skip_recall:
        recall_skip_reason = "flag"
    elif not has_keys:
        recall_skip_reason = "no_keys"
        print(
            "[WARN] ANTHROPIC_API_KEY or VOYAGE_API_KEY not set. "
            "Running structural checks only (--skip-recall implied)."
        )
    else:
        recall_skip_reason = ""

    print(f"Running {len(scenarios)} scenario(s)...")
    if args.verbose:
        print()

    results = []
    for scenario in scenarios:
        print(f"  {scenario.scenario_id}...", end="", flush=True)
        result = await run_scenario(scenario, skip_recall=skip_recall, verbose=args.verbose)
        print(f" {result.overall_score:.3f}  {_status(result.passed)}")
        results.append(result)

    report = aggregate(results, threshold=args.threshold, recall_skip_reason=recall_skip_reason)

    print()
    print(format_report(report))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        def _serializable(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nReport written to {json_path}")

    return 0 if report.passed_threshold else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cairn eval harness — score pipeline quality against scenario ground truth."
    )
    parser.add_argument(
        "--scenarios-dir",
        default="evals/scenarios",
        help="Directory containing scenario YAML files (default: evals/scenarios)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum overall score to pass (default: 0.75)",
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip recall queries (no Voyage AI calls). Structural checks only.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only this scenario_id.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-event and per-query detail for each scenario.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write full report as JSON to this path.",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
