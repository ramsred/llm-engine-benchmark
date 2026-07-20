from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .util import (
    BenchmarkError,
    atomic_write_json,
    coefficient_of_variation,
    ensure_dir,
    load_json,
    median,
    percentile,
    read_jsonl,
    utc_now,
    write_csv,
)


RUN_FIELDS = [
    "engine",
    "cache_mode",
    "concurrency",
    "repetition",
    "valid",
    "sample_count",
    "duration_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "tpot_p50_s",
    "itl_p95_s",
    "e2e_p50_s",
    "e2e_p95_s",
    "request_rps",
    "logical_input_tps",
    "computed_input_tps",
    "computed_input_source",
    "protocol_unique_suffix_tokens_estimate",
    "server_prompt_tokens",
    "server_cached_prompt_tokens",
    "cache_report_coverage_requests",
    "cache_report_coverage_fraction",
    "cache_hit_ratio",
    "output_tps",
    "generated_tokens",
    "run_dir",
]


SUMMARY_FIELDS = [
    "engine",
    "cache_mode",
    "concurrency",
    "valid_repetitions",
    "duration_median_s",
    "duration_min_s",
    "duration_max_s",
    "duration_cv",
    "ttft_p50_median_s",
    "ttft_p95_median_s",
    "tpot_p50_median_s",
    "itl_p95_median_s",
    "e2e_p50_median_s",
    "e2e_p95_median_s",
    "request_rps_median",
    "logical_input_tps_median",
    "computed_input_tps_median",
    "cache_report_coverage_median",
    "server_cached_prompt_tokens_median",
    "cache_hit_ratio_median",
    "output_tps_median",
    "output_tps_min",
    "output_tps_max",
    "output_tps_cv",
]


def generate_report(results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir)
    report_dir = ensure_dir(root / "report")
    runs = _collect_runs(root)
    if not runs:
        raise BenchmarkError(f"No client_results.json files found under {root}")

    matrix_status = _matrix_completeness(root, runs)
    atomic_write_json(report_dir / "report_status.json", matrix_status)

    write_csv(report_dir / "runs.csv", runs, RUN_FIELDS)
    summaries = _summarize_runs(runs)
    write_csv(report_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(
        report_dir / "cold.csv",
        [row for row in summaries if row["cache_mode"] == "cold"],
        SUMMARY_FIELDS,
    )
    write_csv(
        report_dir / "warm_shared.csv",
        [row for row in summaries if row["cache_mode"] == "warm_shared"],
        SUMMARY_FIELDS,
    )
    write_csv(
        report_dir / "exact_repeat.csv",
        [row for row in summaries if row["cache_mode"] == "exact_repeat"],
        SUMMARY_FIELDS,
    )
    by_source = _summarize_by_source(root, runs)
    source_fields = [
        "engine",
        "cache_mode",
        "concurrency",
        "source",
        "requests",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_p50_s",
        "itl_p95_s",
        "e2e_p50_s",
        "e2e_p95_s",
    ]
    write_csv(report_dir / "by_source.csv", by_source, source_fields)

    report_path = report_dir / "report.md"
    report_path.write_text(
        _render_markdown(runs, summaries, by_source, matrix_status), encoding="utf-8"
    )
    return {
        "report": report_path,
        "summary": report_dir / "summary.csv",
        "runs": report_dir / "runs.csv",
        "cold": report_dir / "cold.csv",
        "warm_shared": report_dir / "warm_shared.csv",
        "exact_repeat": report_dir / "exact_repeat.csv",
        "by_source": report_dir / "by_source.csv",
        "status": report_dir / "report_status.json",
    }


def _collect_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active_scope = _active_experiment_scope(root)
    for result_path in sorted(root.glob("*/*/c*/run_*/client_results.json")):
        result = load_json(result_path)
        metadata_path = result_path.parent / "run_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        if active_scope is not None and metadata.get("experiment_scope_signature") != active_scope:
            continue
        rows.append(
            {
                "engine": result.get("engine"),
                "cache_mode": result.get("cache_mode"),
                "concurrency": int(result.get("concurrency", 0)),
                "repetition": int(metadata.get("repetition", _parse_repetition(result_path))),
                "valid": bool(result.get("valid", False)),
                "sample_count": int(result.get("sample_count", 0)),
                "duration_s": result.get("measured_wall_time_seconds"),
                "ttft_p50_s": _nested(result, "ttft_seconds", "p50"),
                "ttft_p95_s": _nested(result, "ttft_seconds", "p95"),
                "tpot_p50_s": _nested(result, "tpot_seconds", "p50"),
                "itl_p95_s": _nested(result, "itl_seconds", "p95"),
                "e2e_p50_s": _nested(result, "e2e_seconds", "p50"),
                "e2e_p95_s": _nested(result, "e2e_seconds", "p95"),
                "request_rps": result.get("request_throughput_per_second"),
                "logical_input_tps": result.get("logical_input_throughput_tokens_per_second"),
                "computed_input_tps": result.get(
                    "computed_input_throughput_tokens_per_second"
                ),
                "computed_input_source": result.get(
                    "computed_input_token_count_source"
                ),
                "protocol_unique_suffix_tokens_estimate": result.get(
                    "protocol_unique_suffix_tokens_estimate"
                ),
                "server_prompt_tokens": result.get(
                    "server_reported_prompt_tokens_total"
                ),
                "server_cached_prompt_tokens": result.get(
                    "server_reported_cached_prompt_tokens_total"
                ),
                "cache_report_coverage_requests": result.get(
                    "cache_report_coverage_requests"
                ),
                "cache_report_coverage_fraction": result.get(
                    "cache_report_coverage_fraction"
                ),
                "cache_hit_ratio": result.get("cache_hit_ratio"),
                "output_tps": result.get("output_throughput_tokens_per_second"),
                "generated_tokens": result.get("generated_output_tokens"),
                "run_dir": str(result_path.parent),
            }
        )
    return rows


def _active_experiment_scope(root: Path) -> str | None:
    path = root / "active_experiment.json"
    if not path.exists():
        return None
    payload = load_json(path)
    value = payload.get("experiment_scope_signature")
    return str(value) if value else None

def _matrix_completeness(root: Path, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active_path = root / "active_experiment.json"
    if not active_path.exists():
        return {
            "scope_defined": False,
            "complete": False,
            "expected_runs": None,
            "observed_runs": len(runs),
            "accepted_runs": sum(1 for row in runs if row["valid"]),
            "missing_runs": [],
            "invalid_runs": [],
            "note": "No active_experiment.json exists; matrix completeness cannot be proven.",
        }

    active = load_json(active_path)
    planned = active.get("runs") if isinstance(active, Mapping) else None
    if not isinstance(planned, list):
        planned = []

    def key(item: Mapping[str, Any]) -> tuple[str, str, int, int]:
        return (
            str(item.get("engine")),
            str(item.get("mode", item.get("cache_mode"))),
            int(item.get("concurrency", 0)),
            int(item.get("repetition", 0)),
        )

    expected = {key(item) for item in planned if isinstance(item, Mapping)}
    observed = {key(row): row for row in runs}
    missing = sorted(expected - set(observed))
    invalid = sorted(item for item in expected if item in observed and not observed[item]["valid"])
    accepted = sum(1 for item in expected if item in observed and observed[item]["valid"])

    def render(item: tuple[str, str, int, int]) -> str:
        engine, mode, concurrency, repetition = item
        return f"{engine}/{mode}/c{concurrency}/run_{repetition:02d}"

    return {
        "scope_defined": True,
        "complete": bool(expected) and not missing and not invalid,
        "expected_runs": len(expected),
        "observed_runs": len(observed),
        "accepted_runs": accepted,
        "missing_runs": [render(item) for item in missing],
        "invalid_runs": [render(item) for item in invalid],
        "note": "Complete only when every planned configuration has one accepted run.",
    }


def _summarize_runs(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        if row["valid"]:
            grouped[(str(row["engine"]), str(row["cache_mode"]), int(row["concurrency"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (engine, mode, concurrency), group in sorted(grouped.items()):
        durations = _numbers(group, "duration_s")
        output_tps = _numbers(group, "output_tps")
        summaries.append(
            {
                "engine": engine,
                "cache_mode": mode,
                "concurrency": concurrency,
                "valid_repetitions": len(group),
                "duration_median_s": median(durations),
                "duration_min_s": min(durations) if durations else None,
                "duration_max_s": max(durations) if durations else None,
                "duration_cv": coefficient_of_variation(durations),
                "ttft_p50_median_s": median(_numbers(group, "ttft_p50_s")),
                "ttft_p95_median_s": median(_numbers(group, "ttft_p95_s")),
                "tpot_p50_median_s": median(_numbers(group, "tpot_p50_s")),
                "itl_p95_median_s": median(_numbers(group, "itl_p95_s")),
                "e2e_p50_median_s": median(_numbers(group, "e2e_p50_s")),
                "e2e_p95_median_s": median(_numbers(group, "e2e_p95_s")),
                "request_rps_median": median(_numbers(group, "request_rps")),
                "logical_input_tps_median": median(_numbers(group, "logical_input_tps")),
                "computed_input_tps_median": median(_numbers(group, "computed_input_tps")),
                "cache_report_coverage_median": median(
                    _numbers(group, "cache_report_coverage_fraction")
                ),
                "server_cached_prompt_tokens_median": median(
                    _numbers(group, "server_cached_prompt_tokens")
                ),
                "cache_hit_ratio_median": median(_numbers(group, "cache_hit_ratio")),
                "output_tps_median": median(output_tps),
                "output_tps_min": min(output_tps) if output_tps else None,
                "output_tps_max": max(output_tps) if output_tps else None,
                "output_tps_cv": coefficient_of_variation(output_tps),
            }
        )
    return summaries


def _summarize_by_source(root: Path, runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tpot": [], "itl": [], "e2e": []}
    )
    counts: dict[tuple[str, str, int, str], int] = defaultdict(int)
    valid_dirs = {Path(str(row["run_dir"])) for row in runs if row["valid"]}
    for run_dir in sorted(valid_dirs):
        client = load_json(run_dir / "client_results.json")
        key_prefix = (
            str(client["engine"]),
            str(client["cache_mode"]),
            int(client["concurrency"]),
        )
        timings_path = run_dir / "request_timings.jsonl"
        if not timings_path.exists():
            continue
        for request in read_jsonl(timings_path):
            if request.get("status") != "ok":
                continue
            source = str(request.get("source") or "unknown")
            key = (*key_prefix, source)
            counts[key] += 1
            _append(groups[key]["ttft"], request.get("ttft_seconds"))
            _append(groups[key]["tpot"], request.get("tpot_seconds"))
            _append(groups[key]["e2e"], request.get("e2e_seconds"))
            groups[key]["itl"].extend(float(value) for value in request.get("itl_seconds", []))

    rows: list[dict[str, Any]] = []
    for (engine, mode, concurrency, source), metrics in sorted(groups.items()):
        rows.append(
            {
                "engine": engine,
                "cache_mode": mode,
                "concurrency": concurrency,
                "source": source,
                "requests": counts[(engine, mode, concurrency, source)],
                "ttft_p50_s": percentile(metrics["ttft"], 50),
                "ttft_p95_s": percentile(metrics["ttft"], 95),
                "tpot_p50_s": percentile(metrics["tpot"], 50),
                "itl_p95_s": percentile(metrics["itl"], 95),
                "e2e_p50_s": percentile(metrics["e2e"], 50),
                "e2e_p95_s": percentile(metrics["e2e"], 95),
            }
        )
    return rows


def _render_markdown(
    runs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    by_source: Sequence[Mapping[str, Any]],
    matrix_status: Mapping[str, Any],
) -> str:
    valid_runs = sum(1 for row in runs if row["valid"])
    invalid_runs = len(runs) - valid_runs
    lines = [
        "# Multi-Backend Long-Context Performance Benchmark Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Accepted runs: **{valid_runs}**  ",
        f"Rejected/invalid runs: **{invalid_runs}**",
        "",
        "",
        f"Matrix status: **{'COMPLETE' if matrix_status.get('complete') else 'INCOMPLETE'}**  ",
        f"Accepted planned runs: **{matrix_status.get('accepted_runs')} / {matrix_status.get('expected_runs')}**",
        "",
        "> This harness measures serving performance and cache behavior. It does not score answer quality; dataset answers are retained only as provenance.",
        "",
        "> The 32-token runtime warm-up avoids benchmark prefixes. Long-context compilation or autotuning on the first measured request is part of observed cold behavior.",
        "",
        "## Aggregate results",
        "",
        _markdown_table(
            summaries,
            [
                ("engine", "Engine"),
                ("cache_mode", "Mode"),
                ("concurrency", "C"),
                ("valid_repetitions", "Runs"),
                ("duration_median_s", "Duration med (s)"),
                ("ttft_p50_median_s", "TTFT P50 med (s)"),
                ("ttft_p95_median_s", "TTFT P95 med (s)"),
                ("tpot_p50_median_s", "TPOT P50 med (s)"),
                ("itl_p95_median_s", "ITL P95 med (s)"),
                ("e2e_p50_median_s", "E2E P50 med (s)"),
                ("e2e_p95_median_s", "E2E P95 med (s)"),
                ("logical_input_tps_median", "Logical input tok/s"),
                ("computed_input_tps_median", "Computed input tok/s"),
                ("cache_report_coverage_median", "Cache report coverage"),
                ("cache_hit_ratio_median", "Cache hit ratio"),
                ("output_tps_median", "Output tok/s"),
            ],
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- Cold and warm observations are kept in separate distributions.",
        "- Warm logical input throughput is cache-assisted prompt acceptance, not raw GPU prefill speed.",
        "- Computed input throughput uses complete server-reported prompt/cache usage details when available. Otherwise cold runs use the fresh-server protocol estimate and warm shared-prefix runs use the known unique-suffix lower bound.",
        "- Cache-hit ratio is shown only when the server reports prompt-token cache details; the coverage column makes missing or partial usage reporting explicit.",
        "- `SGLang --chunked-prefill-size` and `vLLM --max-num-batched-tokens` are matched by intent but are not mechanically identical scheduler controls.",
        "- A run is accepted only when every request succeeds and each request produces the configured fixed output-token count.",
        "- The report uses medians across valid repetitions for aggregate duration and throughput; raw run files retain min/max and per-request timing evidence.",
        "",
        "## Per-source breakdown",
        "",
        _markdown_table(
            by_source,
            [
                ("engine", "Engine"),
                ("cache_mode", "Mode"),
                ("concurrency", "C"),
                ("source", "Source"),
                ("requests", "Requests"),
                ("ttft_p50_s", "TTFT P50 (s)"),
                ("ttft_p95_s", "TTFT P95 (s)"),
                ("tpot_p50_s", "TPOT P50 (s)"),
                ("itl_p95_s", "ITL P95 (s)"),
                ("e2e_p50_s", "E2E P50 (s)"),
                ("e2e_p95_s", "E2E P95 (s)"),
            ],
        ),
        "",
        "## Evidence",
        "",
        "Each run directory contains the neutral client result, per-request timing JSONL, the immutable request-file checksum/reference, server command and image digest, server logs, Prometheus snapshots/deltas, host telemetry, and run metadata.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No valid rows available._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _nested(mapping: Mapping[str, Any], first: str, second: str) -> Any:
    value = mapping.get(first)
    return value.get(second) if isinstance(value, Mapping) else None


def _parse_repetition(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def _append(target: list[float], value: Any) -> None:
    if value is not None:
        target.append(float(value))
