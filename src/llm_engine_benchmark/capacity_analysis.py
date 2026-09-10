from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import BenchmarkError


@dataclass(frozen=True)
class CapacityEvidence:
    offered_rps: float
    repetition: int
    valid: bool
    sla_pass: bool
    achieved_rps: float
    success_fraction: float
    rejection_fraction: float
    queue_p95_seconds: float
    ttft_p95_seconds: float
    itl_p95_seconds: float
    e2e_p95_seconds: float
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _nested_float(value: Mapping[str, Any], *keys: str) -> float:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise BenchmarkError(f"Missing capacity result field: {'.'.join(keys)}")
        current = current[key]
    return float(current)


def _repetition(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("run_"):
        raise BenchmarkError(f"Cannot determine repetition from {path}")
    try:
        return int(name.removeprefix("run_"))
    except ValueError as exc:
        raise BenchmarkError(f"Invalid capacity run directory: {path.parent}") from exc


def _compatibility_record(result: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict:
    model = metadata.get("model") or {}
    return {
        "engine": metadata.get("engine"),
        "cache_mode": metadata.get("cache_mode"),
        "runtime_state": metadata.get("runtime_state"),
        "sample_count": metadata.get("sample_count"),
        "image_digest": metadata.get("image_digest"),
        "model_repo_id": model.get("repo_id"),
        "model_commit_sha": model.get("commit_sha"),
        "sla_targets": (result.get("sla") or {}).get("targets"),
    }


def load_capacity_evidence(
    source_roots: Sequence[Path],
) -> tuple[list[CapacityEvidence], list[Path], dict]:
    rows: list[CapacityEvidence] = []
    inputs: list[Path] = []
    compatibility: dict | None = None
    seen: set[tuple[float, int]] = set()
    for root in source_roots:
        paths = sorted(root.glob("**/capacity_results.json"))
        if not paths:
            raise BenchmarkError(f"No capacity results found: {root}")
        for path in paths:
            metadata_path = path.with_name("capacity_metadata.json")
            if not metadata_path.exists():
                raise BenchmarkError(f"Capacity metadata is missing: {metadata_path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            current = _compatibility_record(result, metadata)
            if compatibility is None:
                compatibility = current
            elif current != compatibility:
                raise BenchmarkError(
                    f"Incompatible capacity campaigns: {path}\n"
                    f"expected={compatibility}\nobserved={current}"
                )
            rate = float(result["offered_request_rate"])
            repetition = _repetition(path)
            key = (rate, repetition)
            if key in seen:
                raise BenchmarkError(
                    f"Duplicate rate/repetition evidence for {rate:g}/run_{repetition:02d}"
                )
            seen.add(key)
            client = result.get("client_metrics") or {}
            rows.append(
                CapacityEvidence(
                    offered_rps=rate,
                    repetition=repetition,
                    valid=bool(result.get("valid")),
                    sla_pass=bool(result.get("sla_pass")),
                    achieved_rps=float(
                        result["achieved_request_throughput_per_second"]
                    ),
                    success_fraction=float(result["success_fraction"]),
                    rejection_fraction=float(result["rejection_fraction"]),
                    queue_p95_seconds=_nested_float(result, "queue_seconds", "p95"),
                    ttft_p95_seconds=_nested_float(client, "ttft_seconds", "p95"),
                    itl_p95_seconds=_nested_float(client, "itl_seconds", "p95"),
                    e2e_p95_seconds=_nested_float(client, "e2e_seconds", "p95"),
                    source=str(path),
                )
            )
            inputs.extend((path, metadata_path))
    return rows, inputs, compatibility or {}


def classify_capacity_evidence(
    rows: Sequence[CapacityEvidence], repetitions: int
) -> dict[str, Any]:
    rates = sorted({row.offered_rps for row in rows})
    rate_statuses: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    for rate in rates:
        rate_rows = [row for row in rows if row.offered_rps == rate]
        valid = [row for row in rate_rows if row.valid]
        passing = [row for row in valid if row.sla_pass]
        failing = [row for row in valid if not row.sla_pass]
        if len(rate_rows) != repetitions or len(valid) != repetitions:
            status = "INCOMPLETE"
        elif len(passing) == repetitions:
            status = "PASS"
        elif len(failing) == repetitions:
            status = "FAIL"
        else:
            status = "UNSTABLE"
        statuses.append(status)
        rate_statuses[f"{rate:g}"] = {
            "status": status,
            "observed_runs": len(rate_rows),
            "valid_runs": len(valid),
            "passing_runs": len(passing),
            "failing_runs": len(failing),
        }

    passing_rates = [rate for rate, status in zip(rates, statuses, strict=True) if status == "PASS"]
    failing_rates = [rate for rate, status in zip(rates, statuses, strict=True) if status == "FAIL"]
    unstable_rates = [
        rate for rate, status in zip(rates, statuses, strict=True) if status == "UNSTABLE"
    ]
    incomplete_rates = [
        rate for rate, status in zip(rates, statuses, strict=True) if status == "INCOMPLETE"
    ]
    non_monotonic = any(
        failing_rate < passing_rate
        for failing_rate in failing_rates
        for passing_rate in passing_rates
    )
    decision_status = "inconclusive"
    lower = max(passing_rates) if passing_rates else None
    upper_candidates = [rate for rate in failing_rates if lower is None or rate > lower]
    upper = min(upper_candidates) if upper_candidates else None
    transition_is_bounded = bool(
        lower is not None
        and upper is not None
        and all(lower < rate < upper for rate in unstable_rates)
    )
    if incomplete_rates:
        decision_status = "inconclusive_incomplete_or_invalid"
    elif non_monotonic:
        decision_status = "inconclusive_non_monotonic"
    elif transition_is_bounded:
        decision_status = "validated_transition_interval"
    elif passing_rates and not failing_rates and not unstable_rates:
        decision_status = "lower_bound_only"
    elif failing_rates and not passing_rates and not unstable_rates:
        decision_status = "below_tested_range"
    elif unstable_rates:
        decision_status = "inconclusive_unstable"

    safe_candidates = [rate for rate in passing_rates if lower is not None and rate < lower]
    recommended = (
        max(safe_candidates)
        if decision_status == "validated_transition_interval" and safe_candidates
        else None
    )
    return {
        "decision_status": decision_status,
        "highest_validated_passing_rps": lower,
        "lowest_validated_failing_rps": upper,
        "recommended_operating_rps": recommended,
        "unstable_rates": unstable_rates,
        "incomplete_rates": incomplete_rates,
        "non_monotonic": non_monotonic,
        "rate_statuses": rate_statuses,
    }


def _mean(rows: Sequence[CapacityEvidence], field: str) -> float:
    return statistics.mean(float(getattr(row, field)) for row in rows)


def _write_csv(path: Path, rows: Sequence[CapacityEvidence]) -> None:
    fields = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.offered_rps, item.repetition)):
            writer.writerow(asdict(row))


def _write_markdown(
    path: Path, rows: Sequence[CapacityEvidence], decision: Mapping[str, Any]
) -> None:
    lines = [
        "# TensorRT-LLM Production Capacity Analysis",
        "",
        "## Repeated SLA validation",
        "",
        (
            "| Offered RPS | Status | Runs | Achieved RPS | Success | Reject | "
            "TTFT P95 | ITL P95 | E2E P95 | Queue P95 |"
        ),
        (
            "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |"
        ),
    ]
    for rate in sorted({row.offered_rps for row in rows}):
        selected = [row for row in rows if row.offered_rps == rate]
        status = decision["rate_statuses"][f"{rate:g}"]["status"]
        lines.append(
            f"| {rate:.6f} | **{status}** | {len(selected)} | "
            f"{_mean(selected, 'achieved_rps'):.6f} | "
            f"{_mean(selected, 'success_fraction'):.3f} | "
            f"{_mean(selected, 'rejection_fraction'):.3f} | "
            f"{_mean(selected, 'ttft_p95_seconds'):.3f} s | "
            f"{_mean(selected, 'itl_p95_seconds'):.3f} s | "
            f"{_mean(selected, 'e2e_p95_seconds'):.3f} s | "
            f"{_mean(selected, 'queue_p95_seconds'):.3f} s |"
        )
    lines.extend(["", "## Decision", ""])
    if decision["decision_status"] == "validated_transition_interval":
        lower = decision["highest_validated_passing_rps"]
        upper = decision["lowest_validated_failing_rps"]
        recommended = decision["recommended_operating_rps"]
        unstable = ", ".join(f"{rate:.6f}" for rate in decision["unstable_rates"])
        lines.extend(
            [
                f"- Highest validated passing load: **{lower:.6f} RPS**.",
                f"- Lowest validated failing load: **{upper:.6f} RPS**.",
                f"- Validated transition interval: **({lower:.6f}, {upper:.6f}) RPS**.",
                f"- Unstable tested rates inside that interval: **{unstable} RPS**.",
            ]
        )
        if recommended is not None:
            lines.append(
                f"- Recommended validated operating point: **{recommended:.6f} RPS**."
            )
        lines.extend(
            [
                "- This is a workload-specific operating envelope, not a universal engine limit.",
                "- Validate burst recovery and N+1 failover before production deployment.",
            ]
        )
    else:
        lines.append(
            f"No production operating envelope is claimed: `{decision['decision_status']}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_capacity_analysis(
    *, source_roots: Sequence[Path], repetitions: int, output_dir: Path
) -> dict[str, Path]:
    if repetitions < 1:
        raise BenchmarkError("repetitions must be at least one")
    rows, inputs, compatibility = load_capacity_evidence(source_roots)
    if not rows:
        raise BenchmarkError("No capacity evidence was loaded")
    decision = classify_capacity_evidence(rows, repetitions)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_dir / "tensorrt-llm-capacity-validation.csv",
        "report": output_dir / "tensorrt-llm-capacity-analysis.md",
        "provenance": output_dir / "tensorrt-llm-capacity-provenance.json",
    }
    _write_csv(outputs["summary"], rows)
    _write_markdown(outputs["report"], rows, decision)
    outputs["provenance"].write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "llm_engine_benchmark.capacity_analysis",
                "benchmark_commit": _git_commit(),
                "source_roots": [str(path) for path in source_roots],
                "source_sha256": {str(path): _sha256(path) for path in sorted(inputs)},
                "compatibility": compatibility,
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate repeated SLA capacity campaigns"
    )
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    args = parser.parse_args(argv)
    try:
        outputs = generate_capacity_analysis(
            source_roots=args.source,
            repetitions=args.repetitions,
            output_dir=args.output_dir,
        )
    except (BenchmarkError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
