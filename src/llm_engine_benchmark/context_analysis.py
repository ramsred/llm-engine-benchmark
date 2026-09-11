from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import BenchmarkError


@dataclass(frozen=True)
class ContextEvidence:
    input_tokens: int
    concurrency: int
    repetition: int
    valid: bool
    successful_requests: int
    cached_prompt_tokens: int
    ttft_p50_seconds: float
    ttft_p95_seconds: float
    itl_p50_seconds: float
    itl_p95_seconds: float
    tpot_mean_seconds: float
    e2e_p95_seconds: float
    output_tok_s: float
    request_rps: float
    max_observed_itl_seconds: float
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


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"Required context artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected a JSON object: {path}")
    return value


def _nested_float(value: Mapping[str, Any], *keys: str) -> float:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise BenchmarkError(f"Missing context result field: {'.'.join(keys)}")
        current = current[key]
    result = float(current)
    if not math.isfinite(result) or result < 0:
        raise BenchmarkError(f"Invalid context metric {'.'.join(keys)}: {result}")
    return result


def _compatibility(plan: Mapping[str, Any]) -> dict[str, Any]:
    options = plan.get("options") or {}
    return {
        "source_sha256": plan.get("source_sha256"),
        "lock": plan.get("lock"),
        "config": plan.get("config"),
        "scope": plan.get("scope"),
        "context_prompt_format_version": plan.get("context_prompt_format_version"),
        "context_measurement_protocol_version": plan.get(
            "context_measurement_protocol_version", 1
        ),
        "samples": options.get("samples"),
        "warmup_waves": options.get("warmup_waves"),
        "profile_nsys": options.get("profile_nsys"),
    }


def _timing_max(path: Path) -> float:
    maximum: float | None = None
    if not path.is_file():
        raise BenchmarkError(f"Context timing evidence is missing: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for gap in row.get("itl_seconds", []):
            value = float(gap)
            if not math.isfinite(value) or value < 0:
                raise BenchmarkError(f"Invalid ITL timing evidence: {path}")
            maximum = value if maximum is None else max(maximum, value)
    if maximum is None:
        raise BenchmarkError(f"No ITL timing evidence found: {path}")
    return maximum


def load_context_evidence(
    source_roots: Sequence[Path],
) -> tuple[list[ContextEvidence], list[Path], dict[str, Any]]:
    rows: list[ContextEvidence] = []
    inputs: list[Path] = []
    compatibility: dict[str, Any] | None = None
    seen: dict[tuple[int, int, int], Path] = {}
    request_hashes: dict[int, str] = {}
    image_digest: str | None = None
    for root in source_roots:
        root = root.resolve()
        plan_path = root / "context_plan.json"
        plan = _read(plan_path)
        current = _compatibility(plan)
        if compatibility is None:
            compatibility = current
        elif current != compatibility:
            raise BenchmarkError(f"Incompatible context experiment source: {root}")
        identity_path = root / "image_identity.json"
        identity = _read(identity_path)
        current_digest = str(identity.get("image_digest") or "")
        if not current_digest:
            raise BenchmarkError(f"Missing image digest: {identity_path}")
        if image_digest is None:
            image_digest = current_digest
        elif current_digest != image_digest:
            raise BenchmarkError(f"Context image digest differs: {root}")
        inputs.extend((plan_path, identity_path))

        for metadata_path in sorted(root.glob("input_*/c*/run_*/context_metadata.json")):
            metadata = _read(metadata_path)
            if metadata.get("status") != "accepted":
                continue
            run_dir = metadata_path.parent
            result_path = run_dir / "client_results.json"
            timings_path = run_dir / "request_timings.jsonl"
            result = _read(result_path)
            tokens = int(metadata["input_tokens"])
            concurrency = int(metadata["concurrency"])
            repetition = int(metadata["repetition"])
            key = (tokens, concurrency, repetition)
            if key in seen:
                raise BenchmarkError(
                    f"Duplicate accepted context cell {key}: {seen[key]} and {run_dir}"
                )
            seen[key] = run_dir
            expected = int(current["samples"])
            if result.get("valid") is not True:
                raise BenchmarkError(f"Accepted context result is invalid: {result_path}")
            if int(result.get("successful_requests", -1)) != expected:
                raise BenchmarkError(f"Context request count differs: {result_path}")
            if int(result.get("failed_requests", 0)) != 0:
                raise BenchmarkError(f"Context result contains failed requests: {result_path}")
            if int(result.get("server_reported_prompt_token_coverage_requests", -1)) != expected:
                raise BenchmarkError(f"Context token-usage coverage differs: {result_path}")
            cached = int(result.get("server_reported_cached_prompt_tokens_total", 0))
            if cached != 0:
                raise BenchmarkError(f"Context cold run contains cached tokens: {result_path}")
            request_hash = str(metadata.get("requests_sha256") or "")
            if not request_hash:
                raise BenchmarkError(f"Context measured prompt hash is missing: {metadata_path}")
            previous_hash = request_hashes.setdefault(tokens, request_hash)
            if previous_hash != request_hash:
                raise BenchmarkError(
                    f"Measured prompts differ between sources at {tokens} tokens"
                )
            if metadata.get("image_digest") != image_digest:
                raise BenchmarkError(f"Context cell image digest differs: {metadata_path}")
            rows.append(
                ContextEvidence(
                    input_tokens=tokens,
                    concurrency=concurrency,
                    repetition=repetition,
                    valid=True,
                    successful_requests=expected,
                    cached_prompt_tokens=cached,
                    ttft_p50_seconds=_nested_float(result, "ttft_seconds", "p50"),
                    ttft_p95_seconds=_nested_float(result, "ttft_seconds", "p95"),
                    itl_p50_seconds=_nested_float(result, "itl_seconds", "p50"),
                    itl_p95_seconds=_nested_float(result, "itl_seconds", "p95"),
                    tpot_mean_seconds=_nested_float(result, "tpot_seconds", "mean"),
                    e2e_p95_seconds=_nested_float(result, "e2e_seconds", "p95"),
                    output_tok_s=_nested_float(
                        result, "output_throughput_tokens_per_second"
                    ),
                    request_rps=_nested_float(result, "request_throughput_per_second"),
                    max_observed_itl_seconds=_timing_max(timings_path),
                    source=str(result_path),
                )
            )
            inputs.extend((metadata_path, result_path, timings_path))
    return rows, inputs, compatibility or {}


def _mean(rows: Sequence[ContextEvidence], field: str) -> float:
    return statistics.mean(float(getattr(row, field)) for row in rows)


def _stdev(rows: Sequence[ContextEvidence], field: str) -> float:
    values = [float(getattr(row, field)) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def classify_context_evidence(
    rows: Sequence[ContextEvidence],
    lengths: Sequence[int],
    concurrencies: Sequence[int],
    repetitions: int,
) -> dict[str, Any]:
    expected = {
        (length, concurrency, repetition)
        for length in lengths
        for concurrency in concurrencies
        for repetition in range(1, repetitions + 1)
    }
    observed = {(row.input_tokens, row.concurrency, row.repetition) for row in rows}
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unexpected:
        raise BenchmarkError(f"Unexpected context cells were loaded: {unexpected}")
    complete = not missing
    return {
        "status": (
            "complete_repeated_validation"
            if complete and repetitions >= 3
            else "complete_discovery"
            if complete
            else "incomplete"
        ),
        "complete": complete,
        "expected_cells": len(expected),
        "accepted_cells": len(observed),
        "missing_cells": [
            {"input_tokens": length, "concurrency": concurrency, "repetition": repetition}
            for length, concurrency, repetition in missing
        ],
    }


def _write_csv(path: Path, rows: Sequence[ContextEvidence]) -> None:
    fields = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (item.input_tokens, item.concurrency, item.repetition)
        ):
            writer.writerow(asdict(row))


def _write_markdown(
    path: Path,
    rows: Sequence[ContextEvidence],
    lengths: Sequence[int],
    concurrencies: Sequence[int],
    repetitions: int,
    decision: Mapping[str, Any],
) -> None:
    lines = [
        "# TensorRT-LLM Context-Scaling Analysis",
        "",
        f"- Evidence status: **{decision['status']}**.",
        f"- Accepted cells: {decision['accepted_cells']} / {decision['expected_cells']}.",
        f"- Repetitions expected per cell: {repetitions}.",
        "- Controlled closed-loop cold-prefix characterization; not production capacity.",
        "",
        "## Context matrix",
        "",
        (
            "| Input tokens | C | Runs | TTFT P95 | ITL P95 | E2E P95 | "
            "Output tok/s |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[int, int], list[ContextEvidence]] = {}
    for row in rows:
        grouped.setdefault((row.input_tokens, row.concurrency), []).append(row)
    for length in sorted(lengths):
        for concurrency in sorted(concurrencies):
            selected = grouped.get((length, concurrency), [])
            if not selected:
                lines.append(f"| {length} | {concurrency} | 0 | — | — | — | — |")
                continue

            def metric(
                field: str, suffix: str = "", selected=selected
            ) -> str:
                mean = _mean(selected, field)
                if len(selected) > 1:
                    return f"{mean:.3f} ± {_stdev(selected, field):.3f}{suffix}"
                return f"{mean:.3f}{suffix}"

            lines.append(
                f"| {length} | {concurrency} | {len(selected)} | "
                f"{metric('ttft_p95_seconds', ' s')} | "
                f"{metric('itl_p95_seconds', ' s')} | "
                f"{metric('e2e_p95_seconds', ' s')} | {metric('output_tok_s')} |"
            )
    lines.extend(["", "## Concurrency effect", ""])
    if 1 in concurrencies and 4 in concurrencies:
        lines.extend(
            [
                "| Input tokens | TTFT C4/C1 | ITL C4/C1 | E2E C4/C1 | Throughput C4/C1 |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for length in sorted(lengths):
            c1 = grouped.get((length, 1), [])
            c4 = grouped.get((length, 4), [])
            if not c1 or not c4:
                continue
            ttft_ratio = _mean(c4, "ttft_p95_seconds") / _mean(c1, "ttft_p95_seconds")
            itl_ratio = _mean(c4, "itl_p95_seconds") / _mean(c1, "itl_p95_seconds")
            e2e_ratio = _mean(c4, "e2e_p95_seconds") / _mean(c1, "e2e_p95_seconds")
            throughput_ratio = _mean(c4, "output_tok_s") / _mean(c1, "output_tok_s")
            lines.append(
                f"| {length} | {ttft_ratio:.2f}× | {itl_ratio:.2f}× | "
                f"{e2e_ratio:.2f}× | {throughput_ratio:.2f}× |"
            )
    lines.extend(["", "## Interpretation guardrails", ""])
    if repetitions == 1:
        lines.append(
            "- This is complete discovery evidence, but one run per cell does not establish "
            "repeatability or confidence intervals."
        )
    if decision["missing_cells"]:
        lines.append(f"- Missing cells: `{json.dumps(decision['missing_cells'])}`.")
    lines.extend(
        [
            (
                "- P95 values are per-run summaries; averaging repeated P95 values is not "
                "a pooled percentile."
            ),
            (
                "- Concurrency ratios describe this fixed workload and do not identify a "
                "kernel-level cause."
            ),
            (
                "- Use separate Nsight runs before attributing changes to attention, KV memory, "
                "or scheduling."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_context_analysis(
    *,
    source_roots: Sequence[Path],
    lengths: Sequence[int],
    concurrencies: Sequence[int],
    repetitions: int,
    output_dir: Path,
) -> dict[str, Path]:
    if repetitions < 1:
        raise BenchmarkError("repetitions must be at least one")
    if not lengths or not concurrencies:
        raise BenchmarkError("lengths and concurrencies must not be empty")
    rows, inputs, compatibility = load_context_evidence(source_roots)
    if not rows:
        raise BenchmarkError("No accepted context evidence was loaded")
    decision = classify_context_evidence(rows, lengths, concurrencies, repetitions)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_dir / "tensorrt-llm-context-scaling-validation.csv",
        "report": output_dir / "tensorrt-llm-context-scaling-analysis.md",
        "provenance": output_dir / "tensorrt-llm-context-scaling-provenance.json",
    }
    _write_csv(outputs["summary"], rows)
    _write_markdown(
        outputs["report"], rows, lengths, concurrencies, repetitions, decision
    )
    outputs["provenance"].write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "llm_engine_benchmark.context_analysis",
                "benchmark_commit": _git_commit(),
                "source_roots": [str(path) for path in source_roots],
                "source_sha256": {str(path): _sha256(path) for path in sorted(set(inputs))},
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


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate context-scaling campaigns")
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--lengths", type=_parse_ints, default=(8000, 32000, 64000, 120000))
    parser.add_argument("--concurrency", type=_parse_ints, default=(1, 4))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    args = parser.parse_args(argv)
    try:
        outputs = generate_context_analysis(
            source_roots=args.source,
            lengths=args.lengths,
            concurrencies=args.concurrency,
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
