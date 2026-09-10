"""Isolated, fixed-length closed-loop characterization, not production capacity.

Only reads the existing lock and canonical manifest. All derived data and runtime
artifacts belong to a newly created output directory; there is no overwrite mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import socket
import string
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .client import ClientRunOptions, run_benchmark_client
from .config import PROJECT_ROOT, load_config, validate_config
from .datasets import _canonical_manifest_reuse_error, manifest_signature
from .environment import capture_environment
from .locking import _validate_lock, _validate_lock_against_config
from .metrics import write_metrics_diff
from .normalize import (
    _build_cold_records,
    encode,
    fit_variable_segment,
    instruction_suffix_with_budget,
    load_pinned_tokenizer,
)
from .orchestrator import select_stratified_records
from .server import DockerEngineServer
from .telemetry import TelemetrySession
from .util import (
    BenchmarkError,
    atomic_write_json,
    atomic_write_text,
    load_json,
    read_jsonl,
    redact_mapping,
    sha256_file,
    sha256_text,
    sha256_token_ids,
    utc_now,
    write_jsonl,
)


@dataclass(frozen=True)
class ContextOptions:
    output_dir: Path
    lengths: tuple[int, ...] = (8000, 32000, 64000, 120000)
    concurrencies: tuple[int, ...] = (1, 4)
    samples: int = 20
    repetitions: int = 1
    warmup_waves: int = 2
    cooldown_seconds: float = 60.0
    skip_image_pull: bool = False
    profile_nsys: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ContextCell:
    input_tokens: int
    concurrency: int
    repetition: int

    @property
    def relative_dir(self) -> Path:
        return Path(f"input_{self.input_tokens}/c{self.concurrency}/run_{self.repetition:02d}")


def cell_config(config: dict, length: int) -> dict:
    derived = copy.deepcopy(config)
    derived["project"]["input_tokens"] = length
    # Shared-prefix preparation is never invoked, but retain a valid config.
    derived["project"]["shared_prefix_tokens"] = min(
        int(config["project"]["shared_prefix_tokens"]), length - 1
    )
    validate_config(derived)
    return derived


def build_context_plan(config: dict, options: ContextOptions) -> list[ContextCell]:
    for name, values in (("lengths", options.lengths), ("concurrencies", options.concurrencies)):
        if not values or len(set(values)) != len(values) or any(v <= 0 for v in values):
            raise BenchmarkError(f"{name} must contain unique positive integers")
    if not max(options.concurrencies) <= options.samples <= 100:
        raise BenchmarkError("samples must be at least the largest concurrency and at most 100")
    if options.repetitions < 1 or options.warmup_waves < 1:
        raise BenchmarkError("repetitions and warmup-waves must be positive")
    if not math.isfinite(options.cooldown_seconds) or options.cooldown_seconds < 0:
        raise BenchmarkError("cooldown-seconds must be finite and nonnegative")
    if max(options.concurrencies) > int(config["engines"]["tensorrt_llm"]["max_batch_size"]):
        raise BenchmarkError("concurrency exceeds configured TensorRT-LLM max_batch_size")
    for length in options.lengths:
        cell_config(config, length)
    pairs = [(length, c) for length in options.lengths for c in options.concurrencies]
    random.Random(int(config["project"]["seed"])).shuffle(pairs)
    return [
        ContextCell(length, concurrency, rep)
        for rep in range(1, options.repetitions + 1)
        for length, concurrency in (pairs if rep % 2 else list(reversed(pairs)))
    ]


def allocate_context_prefixes(tokenizer, sources: list[dict], seed: int) -> dict[str, str]:
    """Allocate prefixes by actual first token, jointly for an entire experiment set.

    Fixed-width decimal markers normally supply many distinct starting tokens.
    Letter fallbacks support tokenizers with different number segmentation. This
    deliberately checks the tokenizer rather than assuming either segmentation.
    Exhaustion fails closed instead of accepting a collision or changing cache policy.
    """
    sample_ids = sorted(str(source["sample_id"]) for source in sources)
    if len(set(sample_ids)) != len(sample_ids):
        raise BenchmarkError("Duplicate sample ID in context prefix allocation")
    markers = iter([*(f"{i:03d}" for i in range(1000)), *string.ascii_letters])
    used: set[int] = set()
    prefixes = {}
    for sample_id in sample_ids:
        nonce = sha256_text(
            json.dumps(
                {
                    "format": "context-prefix-v3",
                    "seed": seed,
                    "sample_id": sample_id,
                },
                sort_keys=True,
            )
        )
        for marker in markers:
            prefix = marker + "\n" + nonce + "\n"
            tokens = encode(tokenizer, prefix)
            if tokens and tokens[0] not in used:
                used.add(tokens[0])
                prefixes[sample_id] = prefix
                break
        else:
            raise BenchmarkError("Tokenizer cannot supply enough distinct first-token prefixes")
    return prefixes


def build_context_records(config: dict, tokenizer, sources: list[dict]) -> list[dict]:
    """Add an early unique prefix without changing the legacy cold-prompt builder.

    The canonical builder starts every prompt with the same descriptive header.
    A unique sample ID later in that header does not prevent reuse of its first
    eight tokens. Even unrelated hashes may share their first token. Allocate
    distinct first tokens before fitting; the caller must include measured and
    warm-up sources together. Refit only the variable body to retain the exact
    length and task suffix, then verify the fitted prompts together.
    """
    records = _build_cold_records(config, tokenizer, sources)
    target = int(config["project"]["input_tokens"])
    prefixes = allocate_context_prefixes(tokenizer, sources, int(config["project"]["seed"]))
    for record, source in zip(records, sources, strict=True):
        suffix, _ = instruction_suffix_with_budget(config, tokenizer, source)
        old_prompt = record["prompt"]
        if not old_prompt.endswith(suffix):
            raise BenchmarkError("Cannot preserve instruction suffix during context prefix refit")
        prompt, tokens, fit = fit_variable_segment(
            tokenizer,
            prefix=prefixes[str(record["sample_id"])],
            segment_source=old_prompt[: -len(suffix)],
            suffix=suffix,
            target_tokens=target,
            label=f"context-prefix:{record['sample_id']}",
        )
        record["prompt"] = prompt
        record["prompt_tokens"] = len(tokens)
        record["metadata"].update(
            {
                "context_prompt_format_version": 3,
                "first_token_id": tokens[0],
                "context_prefix_refit": fit,
                "prompt_sha256": sha256_text(prompt),
                "first_256_token_sha256": sha256_token_ids(tokens[:256]),
            }
        )
    validate_prompt_set(tokenizer, records, target)
    return records


def validate_prompt_set(tokenizer, records: list[dict], target: int) -> dict:
    """Check exact retokenization and reject any shared first token.

    This is protocol evidence, not proof of engine cache hits/misses. Actual
    cached-token usage, when available, is checked separately after execution.
    """
    heads: set[tuple[int, ...]] = set()
    ids: set[str] = set()
    hashes = {}
    for record in records:
        tokens = encode(tokenizer, record["prompt"])
        if len(tokens) != target or record["prompt_tokens"] != target:
            raise BenchmarkError(f"Incorrect token count: {record['sample_id']}")
        head = tuple(tokens[:1])
        if head in heads or record["sample_id"] in ids:
            raise BenchmarkError("Duplicate sample ID or shared first-token prefix")
        heads.add(head)
        ids.add(record["sample_id"])
        hashes[record["sample_id"]] = sha256_text(record["prompt"])
    return {"input_tokens": target, "unique_prefix_check_tokens": 1, "prompt_sha256": hashes}


def load_sources(config: dict) -> tuple[dict, list[dict], dict]:
    lock_path = Path(config["paths"]["lock_file"])
    manifest = Path(config["paths"]["data_dir"]) / "canonical/manifest.jsonl"
    metadata = manifest.with_name("manifest_metadata.json")
    if not all(p.is_file() for p in (lock_path, manifest, metadata)):
        raise BenchmarkError("Existing experiment.lock.json and canonical manifest required")
    lock = load_json(lock_path)
    _validate_lock(lock)
    _validate_lock_against_config(lock, config)
    error = _canonical_manifest_reuse_error(
        manifest, metadata, signature=manifest_signature(config, lock)
    )
    if error:
        raise BenchmarkError(f"Canonical source validation failed: {error}")
    records = list(read_jsonl(manifest))
    if len(records) != 100 or len({r["sample_id"] for r in records}) != 100:
        raise BenchmarkError("Expected 100 unique canonical source records")
    return lock, records, {str(p): sha256_file(p) for p in (lock_path, manifest, metadata)}


def assert_idle(config: dict) -> None:
    """Never remove another benchmark's container or run over an active GPU process."""
    for command in (
        ["docker", "ps", "--filter", "name=llmbench", "--format", "{{.Names}}"],
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip():
            raise BenchmarkError(f"Host is not idle ({command[0]}): {result.stdout.strip()}")
    port = int(config["engines"]["tensorrt_llm"]["host_port"])
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise BenchmarkError(
                f"Port {port} is busy; no existing service will be stopped"
            ) from exc


def accept_result(result: dict, expected: int) -> None:
    if result.get("valid") is not True or result.get("successful_requests") != expected:
        raise BenchmarkError("Client rejected request counts, outputs, or timing measurements")
    if result.get("server_reported_prompt_token_coverage_requests") != expected:
        raise BenchmarkError("Server input-token usage is required for every request")
    if result.get("server_reported_cached_prompt_tokens_total", 0) != 0:
        cached = result["server_reported_cached_prompt_tokens_total"]
        raise BenchmarkError(
            f"Observed prefix-cache reuse in a cold characterization run: {cached} cached tokens"
        )


def observed_concurrency(timings: Path) -> int:
    events = []
    for row in read_jsonl(timings):
        start = float(row["request_start_offset_seconds"])
        end = float(row["request_end_offset_seconds"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise BenchmarkError("Invalid request interval in concurrency evidence")
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def execute_cell(config, lock, tokenizer, cell, options, prepared, digest, root) -> dict:
    run_dir = root / cell.relative_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    derived = cell_config(config, cell.input_tokens)
    # UUID ownership avoids DockerEngineServer's legacy stale-container cleanup
    # ever targeting a container belonging to bench run/capacity or another run.
    derived["engines"]["tensorrt_llm"]["container_name"] = f"llmbench-context-{uuid.uuid4().hex}"
    server = DockerEngineServer(
        engine="tensorrt_llm",
        config=derived,
        lock=lock,
        run_dir=run_dir,
        skip_image_pull=True,
        profile_nsys=options.profile_nsys,
    )
    server.image_digest = digest
    telemetry = TelemetrySession(run_dir, derived.get("telemetry", {}))
    metadata = {
        **asdict(cell),
        "started_at": utc_now(),
        "status": "starting",
        "profile_nsys": options.profile_nsys,
        "context_prompt_format_version": 3,
        "image_digest": digest,
        "warmup_excluded_from_measurement": True,
        "warmup_waves": options.warmup_waves,
        "warmup_requests_per_wave": cell.concurrency,
        "warmup_output_tokens": derived["project"]["output_tokens"],
        "steady_state_limitation": "Representative warm-up, not proof of complete compilation",
        "requests_sha256": sha256_file(prepared / "measured.jsonl"),
        "warmup_sha256": sha256_file(prepared / "warmup.jsonl"),
    }
    atomic_write_json(run_dir / "context_metadata.json", metadata)
    result = None
    try:
        assert_idle(derived)
        server.start()
        server.wait_ready(float(derived["project"]["readiness_timeout_seconds"]))
        metadata["runtime_versions"] = server.capture_runtime_versions()
        metadata["server_command"] = server.run_command
        request_options = ClientRunOptions(
            base_url=server.base_url,
            model=server.api_model,
            engine="tensorrt_llm",
            cache_mode="cold",
            concurrency=cell.concurrency,
            output_tokens=int(derived["project"]["output_tokens"]),
            request_timeout_seconds=float(derived["project"]["request_timeout_seconds"]),
            request_extra=dict(derived["engines"]["tensorrt_llm"].get("request_extra", {})),
            require_server_token_usage=True,
            save_outputs=bool(derived["project"].get("save_outputs", True)),
        )
        metadata["warmup_client_peak_concurrency"] = []
        for wave in range(options.warmup_waves):
            wave_dir = run_dir / "warmup" / f"wave_{wave + 1:02d}"
            warmup = run_benchmark_client(
                records_path=prepared / "warmup.jsonl",
                run_dir=wave_dir,
                tokenizer=tokenizer,
                options=request_options,
                sample_ids=[f"w{wave}-{i}" for i in range(cell.concurrency)],
            )
            accept_result(warmup, cell.concurrency)
            peak = observed_concurrency(wave_dir / "request_timings.jsonl")
            metadata["warmup_client_peak_concurrency"].append(peak)
            if peak != cell.concurrency:
                raise BenchmarkError("Warm-up did not exercise the requested client concurrency")
        server.snapshot_metrics("metrics_before.prom")
        telemetry.start()
        try:
            result = run_benchmark_client(
                records_path=prepared / "measured.jsonl",
                run_dir=run_dir,
                tokenizer=tokenizer,
                options=request_options,
                sample_limit=options.samples,
            )
        finally:
            telemetry.stop()
        server.snapshot_metrics("metrics_after.prom")
        write_metrics_diff(
            run_dir / "metrics_before.prom",
            run_dir / "metrics_after.prom",
            run_dir / "metrics_diff.json",
        )
        accept_result(result, options.samples)
        metadata["measured_client_peak_concurrency"] = observed_concurrency(
            run_dir / "request_timings.jsonl"
        )
        if metadata["measured_client_peak_concurrency"] != cell.concurrency:
            raise BenchmarkError("Measurement did not exercise the requested client concurrency")
        if not server.is_running():
            raise BenchmarkError("Server exited during measurement")
        # Shutdown exports the profile; only accept after its CUDA validation.
        server.stop()
        if options.profile_nsys:
            server.validate_profile_artifacts()
        metadata["status"] = "accepted"
        metadata["cache_evidence"] = (
            "observed_zero_cached_tokens"
            if result.get("cache_report_coverage_fraction") == 1.0
            else "unique_prefix_protocol_only; cache usage not fully reported"
        )
        return result
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if result is not None:
            result["valid"] = False
            result.setdefault("validation_errors", []).append(metadata["error"])
            atomic_write_json(run_dir / "client_results.json", result)
        raise
    finally:
        try:
            telemetry.stop()
        finally:
            try:
                if server.started:
                    server.stop()
            finally:
                metadata["finished_at"] = utc_now()
                atomic_write_json(run_dir / "context_metadata.json", metadata)


def write_context_report(root: Path, plan: list[ContextCell], profiled: bool) -> dict:
    rows = []
    for cell in plan:
        run_dir = root / cell.relative_dir
        metadata_path = run_dir / "context_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        row = {**asdict(cell), "status": metadata.get("status", "not_run")}
        result_path = run_dir / "client_results.json"
        if row["status"] == "accepted" and result_path.is_file():
            result = load_json(result_path)
            for metric, statistic in (
                ("ttft", "p50"),
                ("ttft", "p95"),
                ("itl", "p50"),
                ("itl", "p95"),
                ("tpot", "mean"),
                ("e2e", "p95"),
            ):
                row[f"{metric}_{statistic}_seconds"] = result[f"{metric}_seconds"][statistic]
            row["output_tok_s"] = result["output_throughput_tokens_per_second"]
            row["request_rps"] = result["request_throughput_per_second"]
            row["successful_requests"] = result["successful_requests"]
            timings = read_jsonl(run_dir / "request_timings.jsonl")
            row["max_observed_itl_seconds"] = max(
                (gap for r in timings for gap in r.get("itl_seconds", [])), default=None
            )
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "context_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Context-length characterization",
        "",
        "Controlled closed-loop cold-prefix workload; not production capacity or answer quality.",
        f"Profiled run: {profiled}. Do not combine profiled and unprofiled timings.",
        "Concurrency-matched warm-up is excluded; full runtime stability is not proven.",
        "Server context ceiling, output length and prefill budget are fixed across input lengths.",
        "P95 values are per-run exploratory estimates, not pooled percentiles or P99 evidence.",
        "ITL uses observed SSE timings; multiple tokens may share a timestamp.",
        "Cache protocol checks are not a substitute for measured engine cache-hit evidence.",
        "GPU/memory/thermal telemetry stays in each run directory; missing is not zero.",
        "",
        "| Input tokens | C | Rep | Status | TTFT P95 (s) | ITL P95 (s) | Output tok/s |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:

        def display(key, row=row):
            value = row.get(key)
            return "—" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['input_tokens']} | {row['concurrency']} | {row['repetition']} | "
            f"{row['status']} | {display('ttft_p95_seconds')} | "
            f"{display('itl_p95_seconds')} | {display('output_tok_s')} |"
        )
    atomic_write_text(root / "context_report.md", "\n".join(lines) + "\n")
    summary = {
        "planned_runs": len(plan),
        "accepted_runs": sum(r["status"] == "accepted" for r in rows),
        "finished_at": utc_now(),
        "report": str(root / "context_report.md"),
    }
    atomic_write_json(root / "context_status.json", summary)
    return summary


def run_context_scaling(config: dict, options: ContextOptions) -> dict:
    plan = build_context_plan(config, options)
    if options.dry_run:
        return {
            "dry_run": True,
            "planned_runs": len(plan),
            "runs": [asdict(cell) for cell in plan],
            "output_dir": str(options.output_dir.resolve()),
            "warmup_requests_per_cell": f"{options.warmup_waves} * concurrency",
        }
    root = options.output_dir.resolve()
    if root.exists():
        raise BenchmarkError(
            "Output directory already exists; choose a new directory (no overwrite)"
        )
    for protected in (
        Path(config["paths"]["data_dir"]),
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "tests",
    ):
        if root == protected.resolve() or protected.resolve() in root.parents:
            raise BenchmarkError(
                "Output directory must not be inside canonical data or source code"
            )
    lock, sources, hashes = load_sources(config)
    assert_idle(config)
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": utc_now(),
        "source_sha256": hashes,
        "lock": lock,
        "config": redact_mapping(config),
        "runs": [asdict(cell) for cell in plan],
        "options": {**asdict(options), "output_dir": str(root)},
        "scope": "context_characterization_no_quality_or_production_capacity_claim",
        "context_prompt_format_version": 3,
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, timeout=10
        ).strip()
        payload["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, timeout=10
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        payload["git_commit"] = "unavailable"
    payload["implementation_sha256"] = {
        p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    atomic_write_json(root / "context_plan.json", payload)
    try:
        capture_environment(config, lock, root)
        tokenizer = load_pinned_tokenizer(config, lock)
        selected = select_stratified_records(sources, options.samples)
        warm_sources = [
            {
                "sample_id": f"w{wave}-{i}",
                "source": "synthetic",
                "task": "runtime_warmup",
                "context": "Unrelated initialization document.",
                "instruction": "Return READY.",
                "answer": "READY",
            }
            for wave in range(options.warmup_waves)
            for i in range(max(options.concurrencies))
        ]
        for length in options.lengths:
            derived = cell_config(config, length)
            # One allocation covers ALL prompts sharing a server, including all
            # warm-up waves. Independent allocations could reuse starting tokens.
            combined = build_context_records(derived, tokenizer, selected + warm_sources)
            measured = combined[: len(selected)]
            warmups = combined[len(selected) :]
            checks = validate_prompt_set(tokenizer, measured + warmups, length)
            directory = root / "prepared" / f"input_{length}"
            directory.mkdir(parents=True)
            write_jsonl(directory / "measured.jsonl", measured)
            write_jsonl(directory / "warmup.jsonl", warmups)
            atomic_write_json(directory / "validation.json", checks)
        image_server = DockerEngineServer(
            engine="tensorrt_llm",
            config=config,
            lock=lock,
            run_dir=root / "image",
            skip_image_pull=options.skip_image_pull,
        )
        digest = image_server.prepare_image()
        atomic_write_json(root / "image_identity.json", {"image_digest": digest})
        for index, cell in enumerate(plan):
            print(f"[context] {index + 1:02d}/{len(plan):02d} {cell}", flush=True)
            execute_cell(
                config,
                lock,
                tokenizer,
                cell,
                options,
                root / "prepared" / f"input_{cell.input_tokens}",
                digest,
                root,
            )
            if index + 1 < len(plan):
                time.sleep(options.cooldown_seconds)
    except BaseException as exc:
        atomic_write_json(
            root / "context_failure.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "at": utc_now(),
            },
        )
        raise
    finally:
        write_context_report(root, plan, options.profile_nsys)
        artifacts = {
            str(p.relative_to(root)): sha256_file(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "context_provenance.json"
        }
        atomic_write_json(
            root / "context_provenance.json",
            {
                "source_sha256": hashes,
                "artifact_sha256": artifacts,
                "generated_at": utc_now(),
            },
        )
    return load_json(root / "context_status.json")


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config/tensorrt-llm-context-scaling.yaml")
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=parse_ints, default=(8000, 32000, 64000, 120000))
    parser.add_argument("--concurrency", type=parse_ints, default=(1, 4))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-waves", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--skip-image-pull", action="store_true")
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_context_scaling(
            load_config(args.config),
            ContextOptions(
                output_dir=args.results_dir,
                lengths=args.lengths,
                concurrencies=args.concurrency,
                samples=args.samples,
                repetitions=args.repetitions,
                warmup_waves=args.warmup_waves,
                cooldown_seconds=args.cooldown_seconds,
                skip_image_pull=args.skip_image_pull,
                profile_nsys=args.profile_nsys,
                dry_run=args.dry_run,
            ),
        )
    except (BenchmarkError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0
