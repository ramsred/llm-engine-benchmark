from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .util import BenchmarkError


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    sample_stdev: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class BudgetSummary:
    budget: int
    valid_repetitions: int
    successful_requests: int
    ttft_p95_seconds: MetricSummary
    itl_p95_seconds: MetricSummary
    tpot_mean_seconds: MetricSummary
    e2e_p95_seconds: MetricSummary
    output_throughput_tokens_per_second: MetricSummary


@dataclass(frozen=True)
class ProfileSummary:
    budget: int
    client_itl_p95_seconds: float
    longest_cuda_event_synchronize_seconds: float
    longest_gpu_kernel_milliseconds: float
    longest_gpu_kernel_name: str


METRICS: dict[str, Callable[[dict], float]] = {
    "ttft_p95_seconds": lambda result: float(result["ttft_seconds"]["p95"]),
    "itl_p95_seconds": lambda result: float(result["itl_seconds"]["p95"]),
    "tpot_mean_seconds": lambda result: float(result["tpot_seconds"]["mean"]),
    "e2e_p95_seconds": lambda result: float(result["e2e_seconds"]["p95"]),
    "output_throughput_tokens_per_second": lambda result: float(
        result["output_throughput_tokens_per_second"]
    ),
}


def _metric(values: list[float]) -> MetricSummary:
    return MetricSummary(
        mean=statistics.mean(values),
        sample_stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        values=tuple(values),
    )


def summarize_budget(validation_root: Path, budget: int) -> BudgetSummary:
    run_root = (
        validation_root
        / f"tensorrt-llm-cold-c4-budget-{budget}"
        / "tensorrt_llm/cold/c4"
    )
    result_paths = sorted(run_root.glob("run_[0-9][0-9]/client_results.json"))
    if not result_paths:
        raise BenchmarkError(f"No validation results found for budget {budget}: {run_root}")

    collected = {name: [] for name in METRICS}
    successful_requests = 0
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("valid") is not True:
            raise BenchmarkError(f"Validation result is not accepted: {path}")
        successful_requests += int(result["successful_requests"])
        for name, getter in METRICS.items():
            collected[name].append(getter(result))

    metrics = {name: _metric(values) for name, values in collected.items()}
    return BudgetSummary(
        budget=budget,
        valid_repetitions=len(result_paths),
        successful_requests=successful_requests,
        ttft_p95_seconds=metrics["ttft_p95_seconds"],
        itl_p95_seconds=metrics["itl_p95_seconds"],
        tpot_mean_seconds=metrics["tpot_mean_seconds"],
        e2e_p95_seconds=metrics["e2e_p95_seconds"],
        output_throughput_tokens_per_second=metrics[
            "output_throughput_tokens_per_second"
        ],
    )


def summarize_profile(run_dir: Path, budget: int) -> ProfileSummary:
    result_path = run_dir / "client_results.json"
    database_path = run_dir / "profiling/server.sqlite"
    if not result_path.exists() or not database_path.exists():
        raise BenchmarkError(
            f"Profile {budget} requires client_results.json and profiling/server.sqlite: {run_dir}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise BenchmarkError(f"Profile result is not accepted: {result_path}")

    connection = sqlite3.connect(database_path)
    try:
        sync_row = connection.execute(
            """
            SELECT MAX(runtime.end - runtime.start) / 1000000000.0
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON runtime.nameId = strings.id
            WHERE strings.value LIKE 'cudaEventSynchronize%'
            """
        ).fetchone()
        kernel_row = connection.execute(
            """
            SELECT
                (kernel.end - kernel.start) / 1000000.0,
                strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON kernel.demangledName = strings.id
            ORDER BY kernel.end - kernel.start DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    if not sync_row or sync_row[0] is None or not kernel_row:
        raise BenchmarkError(f"CUDA runtime or kernel evidence is missing: {database_path}")

    return ProfileSummary(
        budget=budget,
        client_itl_p95_seconds=float(result["itl_seconds"]["p95"]),
        longest_cuda_event_synchronize_seconds=float(sync_row[0]),
        longest_gpu_kernel_milliseconds=float(kernel_row[0]),
        longest_gpu_kernel_name=str(kernel_row[1]),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_validation_csv(path: Path, summaries: list[BudgetSummary]) -> None:
    fields = ["budget", "valid_repetitions", "successful_requests"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_sample_stdev"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row: dict[str, object] = {
                "budget": summary.budget,
                "valid_repetitions": summary.valid_repetitions,
                "successful_requests": summary.successful_requests,
            }
            for metric in METRICS:
                value = getattr(summary, metric)
                row[f"{metric}_mean"] = value.mean
                row[f"{metric}_sample_stdev"] = value.sample_stdev
            writer.writerow(row)


def _write_profile_csv(path: Path, profiles: list[ProfileSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(profiles[0])))
        writer.writeheader()
        for profile in profiles:
            writer.writerow(asdict(profile))


def _percent_change(candidate: float, control: float) -> float:
    return ((candidate / control) - 1.0) * 100.0


def _write_markdown(
    path: Path, summaries: list[BudgetSummary], profiles: list[ProfileSummary]
) -> None:
    control = max(summaries, key=lambda row: row.budget)
    latency = min(summaries, key=lambda row: row.itl_p95_seconds.mean)
    balanced = max(summaries, key=lambda row: row.output_throughput_tokens_per_second.mean)
    itl_change = _percent_change(
        latency.itl_p95_seconds.mean, control.itl_p95_seconds.mean
    )
    throughput_change = _percent_change(
        latency.output_throughput_tokens_per_second.mean,
        control.output_throughput_tokens_per_second.mean,
    )
    lines = [
        "# TensorRT-LLM Prefill-Budget Analysis",
        "",
        "## Repeated validation",
        "",
        (
            "| Budget | Valid runs | Requests | TTFT P95 | ITL P95 | TPOT mean | "
            "E2E P95 | Output tok/s |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda value: value.budget, reverse=True):
        lines.append(
            f"| {row.budget} | {row.valid_repetitions} | {row.successful_requests} | "
            f"{row.ttft_p95_seconds.mean:.3f} ± {row.ttft_p95_seconds.sample_stdev:.3f} s | "
            f"{row.itl_p95_seconds.mean:.3f} ± {row.itl_p95_seconds.sample_stdev:.3f} s | "
            f"{row.tpot_mean_seconds.mean:.3f} ± {row.tpot_mean_seconds.sample_stdev:.3f} s | "
            f"{row.e2e_p95_seconds.mean:.3f} ± {row.e2e_p95_seconds.sample_stdev:.3f} s | "
            f"{row.output_throughput_tokens_per_second.mean:.3f} ± "
            f"{row.output_throughput_tokens_per_second.sample_stdev:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Profiler evidence",
            "",
            "| Budget | Client ITL P95 | Longest CUDA synchronization | Longest GPU kernel |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(profiles, key=lambda value: value.budget, reverse=True):
        lines.append(
            f"| {row.budget} | {row.client_itl_p95_seconds:.3f} s | "
            f"{row.longest_cuda_event_synchronize_seconds:.3f} s | "
            f"{row.longest_gpu_kernel_milliseconds:.3f} ms |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Latency-oriented operating point: **{latency.budget} tokens**.",
            f"- Balanced operating point: **{balanced.budget} tokens**.",
            f"- Relative to {control.budget}, the latency-oriented setting changed P95 ITL by "
            f"{itl_change:.1f}% and output throughput by {throughput_change:.1f}%.",
            (
                "- `cudaEventSynchronize` records where the host waited; the preceding GPU "
                "work is the causal bottleneck."
            ),
            "- Conclusions are workload-specific and do not establish a universal engine default.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_prefill_analysis(
    *,
    validation_root: Path,
    budgets: Sequence[int],
    profile_runs: dict[int, Path],
    output_dir: Path,
) -> dict[str, Path]:
    summaries = [summarize_budget(validation_root, budget) for budget in budgets]
    if any(summary.valid_repetitions < 3 for summary in summaries):
        raise BenchmarkError("Each budget requires at least three accepted validation repetitions")
    profiles = [summarize_profile(path, budget) for budget, path in profile_runs.items()]
    if not profiles:
        raise BenchmarkError("At least one --profile BUDGET=RUN_DIR is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation": output_dir / "tensorrt-llm-prefill-budget-validation.csv",
        "profiles": output_dir / "tensorrt-llm-prefill-budget-profiling.csv",
        "report": output_dir / "tensorrt-llm-prefill-budget-analysis.md",
        "provenance": output_dir / "tensorrt-llm-prefill-budget-provenance.json",
    }
    _write_validation_csv(outputs["validation"], summaries)
    _write_profile_csv(outputs["profiles"], profiles)
    _write_markdown(outputs["report"], summaries, profiles)
    validation_inputs = sorted(
        path
        for budget in budgets
        for path in (
            validation_root
            / f"tensorrt-llm-cold-c4-budget-{budget}"
            / "tensorrt_llm/cold/c4"
        ).glob("run_[0-9][0-9]/client_results.json")
    )
    profile_inputs = sorted(
        path
        for run_dir in profile_runs.values()
        for path in (run_dir / "client_results.json", run_dir / "profiling/server.sqlite")
    )
    source_hashes = {
        str(path): _sha256(path) for path in [*validation_inputs, *profile_inputs]
    }
    outputs["provenance"].write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "llm_engine_benchmark.prefill_analysis",
                "benchmark_commit": _git_commit(),
                "validation_root": str(validation_root),
                "budgets": list(budgets),
                "profile_runs": {str(key): str(value) for key, value in profile_runs.items()},
                "source_sha256": source_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def _parse_profile(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", 1)
        return int(budget_text), Path(path_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected BUDGET=RUN_DIR") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze TensorRT-LLM prefill-budget evidence")
    parser.add_argument("--validation-root", type=Path, default=Path("results/validation"))
    parser.add_argument("--budgets", default="8192,4096,2048")
    parser.add_argument("--profile", action="append", type=_parse_profile, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    args = parser.parse_args(argv)

    try:
        budgets = tuple(int(value) for value in args.budgets.split(","))
        outputs = generate_prefill_analysis(
            validation_root=args.validation_root,
            budgets=budgets,
            profile_runs=dict(args.profile),
            output_dir=args.output_dir,
        )
    except (BenchmarkError, ValueError) as exc:
        parser.error(str(exc))
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
