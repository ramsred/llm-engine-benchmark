from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import shutil
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .client import ClientRunOptions, _aggregate_results, _bounded_request, send_warmup_requests
from .environment import capture_environment
from .metrics import write_metrics_diff
from .normalize import (
    encode,
    ensure_token_capacity,
    fit_variable_segment,
    load_pinned_tokenizer,
    preparation_signature,
)
from .orchestrator import (
    _build_runtime_warmup_prompt,
    select_stratified_records,
)
from .server import DockerEngineServer
from .telemetry import TelemetrySession
from .util import (
    BenchmarkError,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    load_json,
    percentile,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_jsonl,
)


@dataclass(frozen=True)
class SlaTargets:
    ttft_p95_seconds: float
    itl_p95_seconds: float
    e2e_p95_seconds: float
    queue_p95_seconds: float
    min_success_fraction: float = 0.99
    max_rejection_fraction: float = 0.01


@dataclass(frozen=True)
class CapacityClientOptions:
    offered_request_rate: float
    max_in_flight: int
    queue_limit: int
    queue_timeout_seconds: float
    arrival_pattern: str
    seed: int
    max_arrival_lag_seconds: float
    sla: SlaTargets


@dataclass(frozen=True)
class CapacityOptions:
    engine: str
    mode: str
    rates: tuple[float, ...]
    requests: int
    repetitions: int
    max_in_flight: int
    queue_limit: int
    queue_timeout_seconds: float
    arrival_pattern: str
    max_arrival_lag_seconds: float
    sla: SlaTargets
    output_dir: Path
    runtime_state: str = "steady"
    runtime_warmup_output_tokens: int = 32
    skip_image_pull: bool = False
    telemetry_enabled: bool = True
    cooldown_seconds: float = 0.0
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class CapacitySpec:
    offered_request_rate: float
    repetition: int
    order_index: int


CAPACITY_RUNTIME_WARMUP_PREFIX = (
    "[CAPACITY-RUNTIME-WARMUP-DO-NOT-CACHE-AS-MEASURED-PREFIX] "
    "This deterministic document exists only to initialize the long-context execution path. "
)


class _AdmissionGate:
    def __init__(self, max_in_flight: int, queue_limit: int) -> None:
        self.max_in_flight = max_in_flight
        self.queue_limit = queue_limit
        self._lock = asyncio.Lock()
        self._waiters: deque[asyncio.Future[None]] = deque()
        self.active = 0
        self.maximum_active = 0
        self.maximum_waiting = 0

    async def acquire(self, timeout_seconds: float) -> tuple[str, float]:
        queued_at = time.perf_counter()
        waiter: asyncio.Future[None] | None = None
        async with self._lock:
            if self.active < self.max_in_flight:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                return "admitted", time.perf_counter() - queued_at
            else:
                if len(self._waiters) >= self.queue_limit:
                    return "rejected_queue_full", 0.0
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.append(waiter)
                self.maximum_waiting = max(self.maximum_waiting, len(self._waiters))

        assert waiter is not None
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout_seconds)
        except TimeoutError:
            async with self._lock:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                else:
                    waiter.cancel()
                    return "rejected_queue_timeout", time.perf_counter() - queued_at
            await waiter
        return "admitted", time.perf_counter() - queued_at

    async def release(self) -> None:
        async with self._lock:
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.done():
                    waiter.set_result(None)
                    return
            self.active -= 1


def generate_arrival_offsets(
    *, count: int, rate: float, pattern: str, seed: int
) -> list[float]:
    if count <= 0:
        raise BenchmarkError("capacity request count must be positive")
    if rate <= 0:
        raise BenchmarkError("offered request rate must be positive")
    if pattern == "constant":
        return [index / rate for index in range(count)]
    if pattern != "poisson":
        raise BenchmarkError(f"Unknown arrival pattern: {pattern}")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, count):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def build_long_context_runtime_warmup(
    *,
    tokenizer,
    target_tokens: int,
    seed: int,
    measured_records: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build an exact-length prompt whose prefix is disjoint from measured traffic."""
    source, source_metadata = ensure_token_capacity(
        tokenizer,
        CAPACITY_RUNTIME_WARMUP_PREFIX,
        required_tokens=target_tokens + 4096,
        seed=seed,
        label="capacity-runtime-warmup",
        min_chars=max(4096, target_tokens * 6),
    )
    prompt, token_ids, fit_metadata = fit_variable_segment(
        tokenizer,
        prefix=CAPACITY_RUNTIME_WARMUP_PREFIX,
        segment_source=source,
        suffix="\nEnd of runtime initialization document. Return only READY.",
        target_tokens=target_tokens,
        label="capacity-runtime-warmup",
    )
    if len(token_ids) != target_tokens:
        raise BenchmarkError(
            f"Long-context runtime warm-up has {len(token_ids)} tokens; "
            f"expected {target_tokens}"
        )

    warmup_head = token_ids[:64]
    maximum_shared_head = 0
    matching_sample: str | None = None
    for record in measured_records:
        measured_head = encode(tokenizer, str(record["prompt"])[:8192])[:64]
        shared = _common_prefix_length(warmup_head, measured_head)
        if shared > maximum_shared_head:
            maximum_shared_head = shared
            matching_sample = str(record["sample_id"])
    if maximum_shared_head >= 16:
        raise BenchmarkError(
            "Long-context runtime warm-up unexpectedly shares a measurable prefix "
            f"({maximum_shared_head} tokens) with {matching_sample}"
        )
    return prompt, {
        "label": "long_context_runtime_warmup",
        "input_tokens": len(token_ids),
        "prompt_sha256": sha256_text(prompt),
        "maximum_shared_prefix_tokens_checked": maximum_shared_head,
        "prefix_check_tokens": len(warmup_head),
        "excluded_from_measurement": True,
        "source": source_metadata,
        "fit": fit_metadata,
    }


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        length += 1
    return length


def run_capacity_client(
    *,
    records: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
    tokenizer,
    request_options: ClientRunOptions,
    capacity_options: CapacityClientOptions,
) -> dict[str, Any]:
    selected = [dict(record) for record in records]
    if not selected:
        raise BenchmarkError("No capacity records were selected")
    return asyncio.run(
        _run_capacity_async(
            records=selected,
            run_dir=Path(run_dir),
            tokenizer=tokenizer,
            request_options=request_options,
            capacity_options=capacity_options,
        )
    )


async def _run_capacity_async(
    *,
    records: list[dict[str, Any]],
    run_dir: Path,
    tokenizer,
    request_options: ClientRunOptions,
    capacity_options: CapacityClientOptions,
) -> dict[str, Any]:
    ensure_dir(run_dir)
    offsets = generate_arrival_offsets(
        count=len(records),
        rate=capacity_options.offered_request_rate,
        pattern=capacity_options.arrival_pattern,
        seed=capacity_options.seed,
    )
    gate = _AdmissionGate(
        capacity_options.max_in_flight,
        capacity_options.queue_limit,
    )
    timeout = aiohttp.ClientTimeout(
        total=request_options.request_timeout_seconds,
        connect=min(60.0, request_options.request_timeout_seconds),
        sock_read=request_options.request_timeout_seconds,
    )
    connector = aiohttp.TCPConnector(
        limit=max(capacity_options.max_in_flight * 2, 8),
        force_close=False,
    )
    wall_origin = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _scheduled_request(
                    session=session,
                    gate=gate,
                    record=record,
                    request_index=index,
                    scheduled_offset=offsets[index],
                    wall_origin=wall_origin,
                    tokenizer=tokenizer,
                    request_options=request_options,
                    queue_timeout_seconds=capacity_options.queue_timeout_seconds,
                )
            )
            for index, record in enumerate(records)
        ]
        results = await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - wall_origin

    results.sort(key=lambda row: int(row["request_index"]))
    serializable: list[dict[str, Any]] = []
    for result in results:
        output_text = str(result.pop("_output_text", ""))
        if request_options.save_outputs and result["status"] == "ok":
            result["output_text"] = output_text
        result["output_sha256"] = sha256_text(output_text)
        serializable.append(result)
    write_jsonl(run_dir / "capacity_request_timings.jsonl", serializable)

    attempted = [row for row in serializable if row["admission_status"] == "admitted"]
    successful = [row for row in attempted if row["status"] == "ok"]
    rejected = [row for row in serializable if row["admission_status"] != "admitted"]
    client_metrics = _aggregate_results(
        successful,
        wall_seconds=wall_seconds,
        options=request_options,
    ) if successful else None
    queue_values = [float(row["queued_seconds"]) for row in attempted]
    arrival_lags = [float(row["arrival_lag_seconds"]) for row in serializable]
    actual_arrival_span = (
        float(serializable[-1]["actual_arrival_offset_seconds"])
        - float(serializable[0]["actual_arrival_offset_seconds"])
        if len(serializable) > 1
        else 0.0
    )
    arrival_lag_p95 = percentile(arrival_lags, 95)
    success_fraction = len(successful) / len(serializable)
    rejection_fraction = len(rejected) / len(serializable)
    client_valid = bool(client_metrics and client_metrics["valid"])
    load_generator_valid = bool(
        arrival_lag_p95 is not None
        and arrival_lag_p95 <= capacity_options.max_arrival_lag_seconds
    )
    result = {
        "created_at": utc_now(),
        "valid": client_valid and load_generator_valid,
        "client_protocol_valid": client_valid,
        "load_generator_valid": load_generator_valid,
        "engine": request_options.engine,
        "cache_mode": request_options.cache_mode,
        "offered_request_rate": capacity_options.offered_request_rate,
        "arrival_pattern": capacity_options.arrival_pattern,
        "scheduled_requests": len(serializable),
        "admitted_requests": len(attempted),
        "successful_requests": len(successful),
        "failed_requests": len(attempted) - len(successful),
        "rejected_requests": len(rejected),
        "queue_full_rejections": sum(
            row["admission_status"] == "rejected_queue_full" for row in rejected
        ),
        "queue_timeout_rejections": sum(
            row["admission_status"] == "rejected_queue_timeout" for row in rejected
        ),
        "success_fraction": success_fraction,
        "rejection_fraction": rejection_fraction,
        "measurement_wall_time_seconds": wall_seconds,
        "scheduled_arrival_span_seconds": offsets[-1] if offsets else 0.0,
        "actual_arrival_span_seconds": actual_arrival_span,
        "actual_offered_request_rate": (
            (len(serializable) - 1) / actual_arrival_span
            if actual_arrival_span > 0
            else None
        ),
        "arrival_lag_seconds": {
            "p50": percentile(arrival_lags, 50),
            "p95": arrival_lag_p95,
            "maximum": max(arrival_lags),
        },
        "max_arrival_lag_seconds": capacity_options.max_arrival_lag_seconds,
        "achieved_request_throughput_per_second": (
            len(successful) / wall_seconds if wall_seconds > 0 else None
        ),
        "max_in_flight": capacity_options.max_in_flight,
        "queue_limit": capacity_options.queue_limit,
        "queue_timeout_seconds": capacity_options.queue_timeout_seconds,
        "maximum_observed_in_flight": gate.maximum_active,
        "maximum_observed_queue_depth": gate.maximum_waiting,
        "queue_seconds": {
            "p50": percentile(queue_values, 50),
            "p95": percentile(queue_values, 95),
            "mean": sum(queue_values) / len(queue_values) if queue_values else None,
        },
        "client_metrics": client_metrics,
    }
    result["sla"] = evaluate_sla(result, capacity_options.sla)
    result["sla_pass"] = bool(result["valid"] and result["sla"]["pass"])
    atomic_write_json(run_dir / "capacity_results.json", result)
    return result


async def _scheduled_request(
    *,
    session: aiohttp.ClientSession,
    gate: _AdmissionGate,
    record: Mapping[str, Any],
    request_index: int,
    scheduled_offset: float,
    wall_origin: float,
    tokenizer,
    request_options: ClientRunOptions,
    queue_timeout_seconds: float,
) -> dict[str, Any]:
    delay = scheduled_offset - (time.perf_counter() - wall_origin)
    if delay > 0:
        await asyncio.sleep(delay)
    arrival_offset = time.perf_counter() - wall_origin
    admission_status, queued_seconds = await gate.acquire(queue_timeout_seconds)
    if admission_status != "admitted":
        return _rejected_result(
            record=record,
            request_index=request_index,
            scheduled_offset=scheduled_offset,
            arrival_offset=arrival_offset,
            admission_status=admission_status,
            queued_seconds=queued_seconds,
        )

    try:
        result = await _bounded_request(
            semaphore=asyncio.Semaphore(1),
            session=session,
            record=record,
            request_index=request_index,
            tokenizer=tokenizer,
            options=request_options,
            wall_origin=wall_origin,
        )
    finally:
        await gate.release()
    result["scheduled_arrival_offset_seconds"] = scheduled_offset
    result["actual_arrival_offset_seconds"] = arrival_offset
    result["arrival_lag_seconds"] = max(0.0, arrival_offset - scheduled_offset)
    result["admission_status"] = "admitted"
    result["queued_seconds"] = queued_seconds
    return result


def _rejected_result(
    *,
    record: Mapping[str, Any],
    request_index: int,
    scheduled_offset: float,
    arrival_offset: float,
    admission_status: str,
    queued_seconds: float,
) -> dict[str, Any]:
    return {
        "request_index": request_index,
        "sample_id": str(record["sample_id"]),
        "source": str(record.get("source") or "unknown"),
        "task": str(record.get("task") or "unknown"),
        "group_id": record.get("group_id"),
        "status": "rejected",
        "error": admission_status,
        "admission_status": admission_status,
        "scheduled_arrival_offset_seconds": scheduled_offset,
        "actual_arrival_offset_seconds": arrival_offset,
        "arrival_lag_seconds": max(0.0, arrival_offset - scheduled_offset),
        "queued_seconds": queued_seconds,
        "request_start_offset_seconds": None,
        "request_end_offset_seconds": arrival_offset,
        "ttft_seconds": None,
        "e2e_seconds": None,
        "tpot_seconds": None,
        "itl_seconds": [],
        "input_tokens": int(record.get("prompt_tokens", 0)),
        "expected_output_tokens": 0,
        "actual_output_tokens": 0,
        "_output_text": "",
    }


def evaluate_sla(result: Mapping[str, Any], targets: SlaTargets) -> dict[str, Any]:
    metrics = result.get("client_metrics")
    observed = {
        "ttft_p95_seconds": _nested_float(metrics, "ttft_seconds", "p95"),
        "itl_p95_seconds": _nested_float(metrics, "itl_seconds", "p95"),
        "e2e_p95_seconds": _nested_float(metrics, "e2e_seconds", "p95"),
        "queue_p95_seconds": _nested_float(result, "queue_seconds", "p95"),
        "success_fraction": float(result.get("success_fraction", 0.0)),
        "rejection_fraction": float(result.get("rejection_fraction", 1.0)),
    }
    limits = asdict(targets)
    checks = {
        "ttft_p95": _at_most(observed["ttft_p95_seconds"], targets.ttft_p95_seconds),
        "itl_p95": _at_most(observed["itl_p95_seconds"], targets.itl_p95_seconds),
        "e2e_p95": _at_most(observed["e2e_p95_seconds"], targets.e2e_p95_seconds),
        "queue_p95": _at_most(observed["queue_p95_seconds"], targets.queue_p95_seconds),
        "success_fraction": observed["success_fraction"] >= targets.min_success_fraction,
        "rejection_fraction": (
            observed["rejection_fraction"] <= targets.max_rejection_fraction
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "targets": limits,
    }


def _nested_float(value: Any, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return float(current) if current is not None else None


def _at_most(observed: float | None, target: float) -> bool:
    return observed is not None and observed <= target


def run_capacity_experiment(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    options: CapacityOptions,
) -> dict[str, Any]:
    _validate_capacity_options(options)
    output_dir = ensure_dir(options.output_dir)
    plan: list[CapacitySpec] = []
    for repetition in range(1, options.repetitions + 1):
        rates = options.rates if repetition % 2 else tuple(reversed(options.rates))
        for rate in rates:
            plan.append(CapacitySpec(rate, repetition, len(plan) + 1))
    plan_payload = {
        "created_at": utc_now(),
        "engine": options.engine,
        "cache_mode": options.mode,
        "requests": options.requests,
        "rates": list(options.rates),
        "repetitions": options.repetitions,
        "admission": {
            "max_in_flight": options.max_in_flight,
            "queue_limit": options.queue_limit,
            "queue_timeout_seconds": options.queue_timeout_seconds,
        },
        "arrival_pattern": options.arrival_pattern,
        "runtime_state": options.runtime_state,
        "runtime_warmup_output_tokens": options.runtime_warmup_output_tokens,
        "sla": asdict(options.sla),
        "runs": [asdict(spec) for spec in plan],
        "preparation_signature": preparation_signature(config, lock),
    }
    atomic_write_json(output_dir / "capacity_run_plan.json", plan_payload)
    if options.dry_run:
        _print_capacity_plan(config, lock, options, plan)
        return {
            "dry_run": True,
            "planned_runs": len(plan),
            "plan": str(output_dir / "capacity_run_plan.json"),
        }

    prepared_dir = Path(config["paths"]["data_dir"]) / "prepared"
    records_path = prepared_dir / (
        "warm_shared.jsonl" if options.mode == "warm_shared" else "cold.jsonl"
    )
    warmup_prefix_path = prepared_dir / "warmup_prefixes.jsonl"
    if not records_path.exists():
        raise BenchmarkError(f"Prepared request file missing: {records_path}")
    all_records = list(read_jsonl(records_path))
    selected_records = select_stratified_records(all_records, options.requests)
    if len(selected_records) != options.requests:
        raise BenchmarkError(
            f"Requested {options.requests} records but found {len(selected_records)}"
        )

    capture_environment(config, lock, output_dir)
    tokenizer = load_pinned_tokenizer(config, lock)
    runtime_warmup_prompt = _build_runtime_warmup_prompt(tokenizer)
    long_context_warmup_prompt: str | None = None
    long_context_warmup_metadata: dict[str, Any] | None = None
    if options.runtime_state == "steady":
        long_context_warmup_prompt, long_context_warmup_metadata = (
            build_long_context_runtime_warmup(
                tokenizer=tokenizer,
                target_tokens=int(config["project"]["input_tokens"]),
                seed=int(config["project"]["seed"]) + 91_173,
                measured_records=selected_records,
            )
        )
    image_probe = DockerEngineServer(
        engine=options.engine,
        config=config,
        lock=lock,
        run_dir=output_dir / "environment" / f"image_{options.engine}",
        skip_image_pull=options.skip_image_pull,
    )
    image_digest = image_probe.prepare_image()
    atomic_write_json(
        output_dir / "environment" / "docker_images.json",
        {options.engine: image_digest},
    )
    plan_payload["image_digest"] = image_digest
    experiment_scope_signature = sha256_text(json.dumps(plan_payload, sort_keys=True))
    plan_payload["experiment_scope_signature"] = experiment_scope_signature
    atomic_write_json(output_dir / "capacity_run_plan.json", plan_payload)

    completed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    for index, spec in enumerate(plan):
        run_dir = _capacity_run_dir(output_dir, options, spec)
        signature = _capacity_signature(config, lock, options, spec, image_digest)
        action = _prepare_capacity_run_directory(
            run_dir,
            signature=signature,
            experiment_scope_signature=experiment_scope_signature,
            resume=options.resume,
            overwrite=options.overwrite,
        )
        if action == "skip":
            skipped += 1
            continue
        print(
            f"[capacity] {spec.order_index:02d}/{len(plan):02d} "
            f"rate={spec.offered_request_rate:g} repetition={spec.repetition}"
        )
        try:
            _execute_capacity_run(
                config=config,
                lock=lock,
                tokenizer=tokenizer,
                runtime_warmup_prompt=runtime_warmup_prompt,
                long_context_warmup_prompt=long_context_warmup_prompt,
                long_context_warmup_metadata=long_context_warmup_metadata,
                records_path=records_path,
                selected_records=selected_records,
                warmup_prefix_path=warmup_prefix_path,
                image_digest=image_digest,
                run_dir=run_dir,
                signature=signature,
                experiment_scope_signature=experiment_scope_signature,
                spec=spec,
                options=options,
            )
            completed += 1
        except Exception as exc:
            failure = {
                "rate": spec.offered_request_rate,
                "repetition": spec.repetition,
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            atomic_write_json(run_dir / "capacity_failure.json", failure)
            raise
        finally:
            if index < len(plan) - 1 and options.cooldown_seconds > 0:
                time.sleep(options.cooldown_seconds)

    report = generate_capacity_report(output_dir, plan, options)
    summary = {
        "finished_at": utc_now(),
        "planned_runs": len(plan),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "failed_runs": len(failures),
        "failures": failures,
        **report,
    }
    atomic_write_json(output_dir / "capacity_experiment.json", summary)
    return summary


def _execute_capacity_run(
    *,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    tokenizer,
    runtime_warmup_prompt: str,
    long_context_warmup_prompt: str | None,
    long_context_warmup_metadata: Mapping[str, Any] | None,
    records_path: Path,
    selected_records: list[dict[str, Any]],
    warmup_prefix_path: Path,
    image_digest: str,
    run_dir: Path,
    signature: str,
    experiment_scope_signature: str,
    spec: CapacitySpec,
    options: CapacityOptions,
) -> None:
    ensure_dir(run_dir)
    atomic_write_json(
        run_dir / "requests.reference.json",
        {
            "prepared_file": str(records_path.resolve()),
            "prepared_file_sha256": sha256_file(records_path),
            "selected_sample_ids": [row["sample_id"] for row in selected_records],
        },
    )
    metadata = {
        "run_signature": signature,
        "experiment_scope_signature": experiment_scope_signature,
        "started_at": utc_now(),
        "benchmark_scope": "open_loop_sla_capacity",
        "engine": options.engine,
        "cache_mode": options.mode,
        "offered_request_rate": spec.offered_request_rate,
        "repetition": spec.repetition,
        "sample_count": len(selected_records),
        "image": config["engines"][options.engine]["image"],
        "image_digest": image_digest,
        "model": lock["model"],
        "capacity_options": _capacity_option_payload(options),
        "runtime_state": options.runtime_state,
        "runtime_warmup_tokens": len(encode(tokenizer, runtime_warmup_prompt)),
        "runtime_warmup_sha256": sha256_text(runtime_warmup_prompt),
        "long_context_runtime_warmup": long_context_warmup_metadata,
        "runtime_warmup_limitation": (
            "An exact representative long-context request initializes steady-state "
            "execution paths before measurement and is excluded from results."
            if options.runtime_state == "steady"
            else "Only the unrelated 32-token runtime warm-up runs before measurement; "
            "first-request long-context initialization remains measured by design."
        ),
        "status": "starting",
    }
    atomic_write_json(run_dir / "capacity_metadata.json", metadata)
    server = DockerEngineServer(
        engine=options.engine,
        config=config,
        lock=lock,
        run_dir=run_dir,
        skip_image_pull=True,
    )
    server.image_digest = image_digest
    telemetry_config = dict(config.get("telemetry", {}))
    telemetry_config["enabled"] = bool(
        options.telemetry_enabled and telemetry_config.get("enabled", True)
    )
    telemetry = TelemetrySession(run_dir, telemetry_config)
    engine_config = config["engines"][options.engine]
    request_extra = dict(engine_config.get("request_extra", {}))
    try:
        server.start()
        metadata["server_command"] = server.run_command
        metadata["status"] = "server_started"
        atomic_write_json(run_dir / "capacity_metadata.json", metadata)
        server.wait_ready(float(config["project"]["readiness_timeout_seconds"]))
        metadata["runtime_versions"] = server.capture_runtime_versions()
        atomic_write_json(run_dir / "capacity_metadata.json", metadata)

        warmup_results = send_warmup_requests(
            base_url=server.base_url,
            model=server.api_model,
            prompts=[("short_runtime_warmup", runtime_warmup_prompt)],
            tokenizer=tokenizer,
            request_extra=request_extra,
            timeout_seconds=float(config["project"]["request_timeout_seconds"]),
        )
        long_context_result: dict[str, Any] | None = None
        if long_context_warmup_prompt is not None:
            long_context_result = send_warmup_requests(
                base_url=server.base_url,
                model=server.api_model,
                prompts=[("long_context_runtime_warmup", long_context_warmup_prompt)],
                tokenizer=tokenizer,
                request_extra=request_extra,
                timeout_seconds=float(config["project"]["request_timeout_seconds"]),
                max_tokens=options.runtime_warmup_output_tokens,
            )[0]
            expected_input = int(config["project"]["input_tokens"])
            if long_context_result["input_tokens"] != expected_input:
                raise BenchmarkError(
                    "Representative runtime warm-up input-token mismatch: "
                    f"{long_context_result['input_tokens']} != {expected_input}"
                )
            if long_context_result["output_tokens"] != options.runtime_warmup_output_tokens:
                raise BenchmarkError(
                    "Representative runtime warm-up output-token mismatch: "
                    f"{long_context_result['output_tokens']} != "
                    f"{options.runtime_warmup_output_tokens}"
                )
            atomic_write_json(
                run_dir / "long_context_warmup.json",
                {**dict(long_context_warmup_metadata or {}), "result": long_context_result},
            )
            warmup_results.append(long_context_result)
        prefix_warmups: list[tuple[str, str]] = []
        if options.mode == "warm_shared":
            selected_groups = {
                str(row["group_id"])
                for row in selected_records
                if row.get("group_id") is not None
            }
            warmup_by_group = {
                str(row["group_id"]): str(row["prompt"])
                for row in read_jsonl(warmup_prefix_path)
            }
            missing = selected_groups - set(warmup_by_group)
            if missing:
                raise BenchmarkError(f"Missing warm-up prompts for groups: {sorted(missing)}")
            prefix_warmups.extend(
                (f"prefix_warmup:{group}", warmup_by_group[group])
                for group in sorted(selected_groups)
            )
        if prefix_warmups:
            warmup_results.extend(
                send_warmup_requests(
                    base_url=server.base_url,
                    model=server.api_model,
                    prompts=prefix_warmups,
                    tokenizer=tokenizer,
                    request_extra=request_extra,
                    timeout_seconds=float(config["project"]["request_timeout_seconds"]),
                )
            )
        atomic_write_json(
            run_dir / "warmup_results.json",
            {
                "cache_mode": options.mode,
                "runtime_state": options.runtime_state,
                "completed_at": utc_now(),
                "results": warmup_results,
            },
        )
        server.snapshot_metrics("metrics_before.prom")
        telemetry.start()
        try:
            request_options = ClientRunOptions(
                base_url=server.base_url,
                model=server.api_model,
                engine=options.engine,
                cache_mode=options.mode,
                concurrency=options.max_in_flight,
                output_tokens=int(config["project"]["output_tokens"]),
                request_timeout_seconds=float(config["project"]["request_timeout_seconds"]),
                request_extra=request_extra,
                save_outputs=bool(config["project"].get("save_outputs", True)),
                require_server_token_usage=bool(
                    config["project"].get("require_server_token_usage", True)
                ),
            )
            result = run_capacity_client(
                records=selected_records,
                run_dir=run_dir,
                tokenizer=tokenizer,
                request_options=request_options,
                capacity_options=CapacityClientOptions(
                    offered_request_rate=spec.offered_request_rate,
                    max_in_flight=options.max_in_flight,
                    queue_limit=options.queue_limit,
                    queue_timeout_seconds=options.queue_timeout_seconds,
                    arrival_pattern=options.arrival_pattern,
                    seed=int(config["project"]["seed"]) + spec.repetition,
                    max_arrival_lag_seconds=options.max_arrival_lag_seconds,
                    sla=options.sla,
                ),
            )
        finally:
            telemetry.stop()
        server.snapshot_metrics("metrics_after.prom")
        write_metrics_diff(
            run_dir / "metrics_before.prom",
            run_dir / "metrics_after.prom",
            run_dir / "metrics_diff.json",
        )
        if not server.is_running():
            result["valid"] = False
            result["sla_pass"] = False
            result["server_running_after_measurement"] = False
            atomic_write_json(run_dir / "capacity_results.json", result)
        else:
            result["server_running_after_measurement"] = True
            atomic_write_json(run_dir / "capacity_results.json", result)
        metadata["finished_at"] = utc_now()
        metadata["status"] = "measured"
        metadata["measurement_valid"] = result["valid"]
        metadata["sla_pass"] = result["sla_pass"]
        atomic_write_json(run_dir / "capacity_metadata.json", metadata)
    finally:
        try:
            telemetry.stop()
        except Exception:
            pass
        server.stop()


def generate_capacity_report(
    output_dir: Path,
    plan: Sequence[CapacitySpec],
    options: CapacityOptions,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in plan:
        path = _capacity_run_dir(output_dir, options, spec) / "capacity_results.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        client = result.get("client_metrics") or {}
        rows.append(
            {
                "offered_request_rate": result["offered_request_rate"],
                "repetition": spec.repetition,
                "valid": result["valid"],
                "sla_pass": result["sla_pass"],
                "scheduled_requests": result["scheduled_requests"],
                "successful_requests": result["successful_requests"],
                "rejected_requests": result["rejected_requests"],
                "success_fraction": result["success_fraction"],
                "rejection_fraction": result["rejection_fraction"],
                "achieved_request_throughput_per_second": result[
                    "achieved_request_throughput_per_second"
                ],
                "actual_offered_request_rate": result["actual_offered_request_rate"],
                "arrival_lag_p95_seconds": result["arrival_lag_seconds"]["p95"],
                "queue_p95_seconds": result["queue_seconds"]["p95"],
                "ttft_p95_seconds": _nested_float(client, "ttft_seconds", "p95"),
                "itl_p95_seconds": _nested_float(client, "itl_seconds", "p95"),
                "e2e_p95_seconds": _nested_float(client, "e2e_seconds", "p95"),
            }
        )
    csv_path = output_dir / "capacity_summary.csv"
    fields = list(rows[0]) if rows else ["offered_request_rate", "repetition"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    decision = classify_capacity_rates(rows, options.rates, options.repetitions)
    boundary = decision["capacity_boundary_rps"]
    safe_capacity = decision["recommended_safe_capacity_rps"]
    report_path = output_dir / "capacity_report.md"
    lines = [
        "# SLA-Constrained Capacity Report",
        "",
        f"- Engine: `{options.engine}`",
        f"- Cache mode: `{options.mode}`",
        f"- Arrival pattern: `{options.arrival_pattern}`",
        f"- Requests per run: {options.requests}",
        f"- Repetitions per rate: {options.repetitions}",
        f"- Runtime state: `{options.runtime_state}`",
        f"- P95 TTFT / ITL / E2E SLA: {options.sla.ttft_p95_seconds:g} / "
        f"{options.sla.itl_p95_seconds:g} / {options.sla.e2e_p95_seconds:g} seconds",
        f"- P95 admission queue SLA: {options.sla.queue_p95_seconds:g} seconds",
        "",
        (
            "| Offered RPS | Achieved RPS | Rep | Valid | SLA | Success | Reject | "
            "Arrival lag P95 | TTFT P95 | ITL P95 | E2E P95 | Queue P95 |"
        ),
        (
            "| ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for row in sorted(rows, key=lambda value: (value["offered_request_rate"], value["repetition"])):
        lines.append(
            f"| {row['offered_request_rate']:.6f} | "
            f"{_format_rate(row['achieved_request_throughput_per_second'])} | "
            f"{row['repetition']} | "
            f"{'PASS' if row['valid'] else 'FAIL'} | "
            f"{'PASS' if row['sla_pass'] else 'FAIL'} | "
            f"{row['success_fraction']:.3f} | {row['rejection_fraction']:.3f} | "
            f"{_format_seconds(row['arrival_lag_p95_seconds'])} | "
            f"{_format_seconds(row['ttft_p95_seconds'])} | "
            f"{_format_seconds(row['itl_p95_seconds'])} | "
            f"{_format_seconds(row['e2e_p95_seconds'])} | "
            f"{_format_seconds(row['queue_p95_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## Rate classification",
            "",
            "| Offered RPS | Status | Valid | Pass | Fail |",
            "| ---: | :---: | ---: | ---: | ---: |",
        ]
    )
    for rate in sorted(options.rates):
        item = decision["rate_statuses"][_rate_key(rate)]
        lines.append(
            f"| {rate:.6f} | **{item['status']}** | {item['valid_runs']} | "
            f"{item['passing_runs']} | {item['failing_runs']} |"
        )
    lines.extend(["", "## Decision", ""])
    if decision["decision_status"] == "validated_boundary":
        lines.extend(
            [
                f"- Validated capacity boundary: **{boundary:.6f} requests/second**.",
                "- First stable failing rate: "
                f"**{decision['next_failing_rate_rps']:.6f} requests/second**.",
                (
                    "- Suggested 25% headroom operating point: "
                    f"**{safe_capacity:.6f} requests/second**."
                ),
                "- Validate burst recovery and N+1 failover before production deployment.",
            ]
        )
    elif decision["decision_status"] == "lower_bound_only":
        lines.append(
            "All tested rates passed. Capacity is **at least "
            f"{decision['capacity_lower_bound_rps']:.6f} RPS**; "
            "test a higher rate before claiming a boundary or applying a headroom recommendation."
        )
    elif decision["decision_status"] == "below_tested_range":
        lines.append(
            "The lowest tested rate failed consistently. Capacity is below the tested range; "
            "add lower rates."
        )
    else:
        lines.append(
            "No capacity boundary is claimed because the rate sweep is incomplete, invalid, "
            "unstable, or non-monotonic. Repeat the identified rates before making a "
            "production decision."
        )
        if decision["rates_to_repeat"]:
            rendered = ", ".join(f"{rate:.6f}" for rate in decision["rates_to_repeat"])
            lines.append(f"- Rates to repeat: {rendered} RPS.")
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return {
        **decision,
        "summary_csv": str(csv_path),
        "report": str(report_path),
    }


def classify_capacity_rates(
    rows: Sequence[Mapping[str, Any]],
    rates: Sequence[float],
    repetitions: int,
) -> dict[str, Any]:
    """Classify an ordered sweep without inventing a boundary from noisy evidence."""
    ordered_rates = sorted(rates)
    rate_statuses: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    for rate in ordered_rates:
        rate_rows = [row for row in rows if float(row["offered_request_rate"]) == rate]
        valid_rows = [row for row in rate_rows if bool(row.get("valid"))]
        passing = [row for row in valid_rows if bool(row.get("sla_pass"))]
        failing = [row for row in valid_rows if not bool(row.get("sla_pass"))]
        if len(rate_rows) != repetitions:
            status = "INCOMPLETE"
        elif len(valid_rows) != repetitions:
            status = "INVALID"
        elif len(passing) == repetitions:
            status = "PASS"
        elif len(failing) == repetitions:
            status = "FAIL"
        else:
            status = "UNSTABLE"
        statuses.append(status)
        rate_statuses[_rate_key(rate)] = {
            "status": status,
            "observed_runs": len(rate_rows),
            "expected_runs": repetitions,
            "valid_runs": len(valid_rows),
            "passing_runs": len(passing),
            "failing_runs": len(failing),
        }

    initial_pass_count = 0
    for status in statuses:
        if status != "PASS":
            break
        initial_pass_count += 1
    highest_contiguous = (
        ordered_rates[initial_pass_count - 1] if initial_pass_count else None
    )
    boundary: float | None = None
    next_failing: float | None = None
    lower_bound: float | None = None
    decision_status: str
    repeat_rates: list[float] = []

    if statuses and all(status == "PASS" for status in statuses):
        decision_status = "lower_bound_only"
        lower_bound = ordered_rates[-1]
    elif any(status in {"INCOMPLETE", "INVALID"} for status in statuses):
        decision_status = "inconclusive_incomplete_or_invalid"
        repeat_rates = [
            rate for rate, status in zip(ordered_rates, statuses, strict=True)
            if status in {"INCOMPLETE", "INVALID"}
        ]
    elif any(
        status == "PASS" for status in statuses[initial_pass_count + 1 :]
    ):
        decision_status = "inconclusive_non_monotonic"
        repeat_rates = [
            rate for rate, status in zip(ordered_rates, statuses, strict=True)
            if status != "PASS"
        ] + [
            rate
            for rate, status in zip(
                ordered_rates[initial_pass_count + 1 :],
                statuses[initial_pass_count + 1 :],
                strict=True,
            )
            if status == "PASS"
        ]
        repeat_rates = sorted(set(repeat_rates))
    elif "UNSTABLE" in statuses:
        decision_status = "inconclusive_unstable"
        repeat_rates = [
            rate
            for rate, status in zip(ordered_rates, statuses, strict=True)
            if status == "UNSTABLE"
        ]
    elif statuses and statuses[0] == "FAIL":
        decision_status = "below_tested_range"
        next_failing = ordered_rates[0]
    elif initial_pass_count and statuses[initial_pass_count] == "FAIL":
        decision_status = "validated_boundary"
        boundary = ordered_rates[initial_pass_count - 1]
        next_failing = ordered_rates[initial_pass_count]
    else:
        decision_status = "inconclusive"

    return {
        "decision_status": decision_status,
        "capacity_boundary_rps": boundary,
        "capacity_lower_bound_rps": lower_bound,
        "highest_contiguous_passing_rps": highest_contiguous,
        "next_failing_rate_rps": next_failing,
        "recommended_safe_capacity_rps": boundary * 0.75 if boundary is not None else None,
        "non_monotonic": decision_status == "inconclusive_non_monotonic",
        "rates_to_repeat": repeat_rates,
        "rate_statuses": rate_statuses,
    }


def _rate_key(rate: float) -> str:
    return f"{rate:.12g}"


def _format_seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f} s"


def _format_rate(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def _capacity_run_dir(
    output_dir: Path, options: CapacityOptions, spec: CapacitySpec
) -> Path:
    rate_label = f"{spec.offered_request_rate:.6f}".replace(".", "p")
    return (
        output_dir
        / options.engine
        / options.mode
        / f"rps_{rate_label}"
        / f"run_{spec.repetition:02d}"
    )


def _prepare_capacity_run_directory(
    run_dir: Path,
    *,
    signature: str,
    experiment_scope_signature: str,
    resume: bool,
    overwrite: bool,
) -> str:
    if run_dir.exists():
        metadata_path = run_dir / "capacity_metadata.json"
        results_path = run_dir / "capacity_results.json"
        if resume and metadata_path.exists() and results_path.exists():
            metadata = load_json(metadata_path)
            results = load_json(results_path)
            if metadata.get("run_signature") == signature and results.get("valid") is True:
                previous_scope = metadata.get("experiment_scope_signature")
                if previous_scope != experiment_scope_signature:
                    source_scopes = metadata.get("source_experiment_scope_signatures", [])
                    if not isinstance(source_scopes, list):
                        source_scopes = []
                    if previous_scope and previous_scope not in source_scopes:
                        source_scopes.append(previous_scope)
                    metadata["source_experiment_scope_signatures"] = source_scopes
                    metadata["experiment_scope_signature"] = experiment_scope_signature
                    metadata["resumed_into_scope_at"] = utc_now()
                    atomic_write_json(metadata_path, metadata)
                return "skip"
        if resume and not any(run_dir.iterdir()):
            return "run"
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise BenchmarkError(
                f"Capacity run directory already exists: {run_dir}. "
                "Use --resume or --overwrite."
            )
    ensure_dir(run_dir)
    return "run"


def _capacity_signature(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    options: CapacityOptions,
    spec: CapacitySpec,
    image_digest: str,
) -> str:
    return sha256_text(
        json.dumps(
            {
                "preparation_signature": preparation_signature(config, lock),
                "image_digest": image_digest,
                "engine": options.engine,
                "mode": options.mode,
                "rate": spec.offered_request_rate,
                "repetition": spec.repetition,
                "options": _capacity_option_payload(options),
            },
            sort_keys=True,
        )
    )


def _capacity_option_payload(options: CapacityOptions) -> dict[str, Any]:
    return {
        "requests": options.requests,
        "runtime_state": options.runtime_state,
        "runtime_warmup_output_tokens": options.runtime_warmup_output_tokens,
        "max_in_flight": options.max_in_flight,
        "queue_limit": options.queue_limit,
        "queue_timeout_seconds": options.queue_timeout_seconds,
        "arrival_pattern": options.arrival_pattern,
        "max_arrival_lag_seconds": options.max_arrival_lag_seconds,
        "sla": asdict(options.sla),
    }


def _validate_capacity_options(options: CapacityOptions) -> None:
    if options.engine not in {"vllm", "sglang", "tensorrt_llm"}:
        raise BenchmarkError(f"Unsupported capacity engine: {options.engine}")
    if options.mode not in {"cold", "warm_shared"}:
        raise BenchmarkError("Capacity mode must be cold or warm_shared")
    if options.runtime_state not in {"steady", "cold-start"}:
        raise BenchmarkError("Runtime state must be steady or cold-start")
    if options.runtime_warmup_output_tokens <= 0:
        raise BenchmarkError("Runtime warm-up output tokens must be positive")
    if not options.rates or any(rate <= 0 or not math.isfinite(rate) for rate in options.rates):
        raise BenchmarkError("Capacity rates must be finite positive numbers")
    if len(set(options.rates)) != len(options.rates):
        raise BenchmarkError("Capacity rates must not contain duplicates")
    if not 1 <= options.requests <= 100:
        raise BenchmarkError("Capacity requests must be between 1 and 100")
    if options.repetitions <= 0 or options.max_in_flight <= 0:
        raise BenchmarkError("Repetitions and max-in-flight must be positive")
    if options.queue_limit < 0 or options.queue_timeout_seconds <= 0:
        raise BenchmarkError("Queue limit must be nonnegative and timeout must be positive")
    if options.arrival_pattern not in {"constant", "poisson"}:
        raise BenchmarkError("Arrival pattern must be constant or poisson")
    if options.max_arrival_lag_seconds <= 0:
        raise BenchmarkError("Maximum arrival lag must be positive")
    for name, value in asdict(options.sla).items():
        if not math.isfinite(value):
            raise BenchmarkError(f"SLA target {name} must be finite")
    if min(
        options.sla.ttft_p95_seconds,
        options.sla.itl_p95_seconds,
        options.sla.e2e_p95_seconds,
    ) <= 0 or options.sla.queue_p95_seconds < 0:
        raise BenchmarkError("Latency SLA targets must be positive; queue P95 may be zero")
    if not 0 <= options.sla.min_success_fraction <= 1:
        raise BenchmarkError("Minimum success fraction must be between 0 and 1")
    if not 0 <= options.sla.max_rejection_fraction <= 1:
        raise BenchmarkError("Maximum rejection fraction must be between 0 and 1")


def _print_capacity_plan(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    options: CapacityOptions,
    plan: Sequence[CapacitySpec],
) -> None:
    print(f"Capacity dry-run plan: {len(plan)} fresh-server runs")
    for spec in plan:
        run_dir = _capacity_run_dir(options.output_dir, options, spec)
        server = DockerEngineServer(
            engine=options.engine,
            config=config,
            lock=lock,
            run_dir=run_dir,
            skip_image_pull=True,
        )
        command = server._build_docker_command(server._build_server_args())
        print(
            f"{spec.order_index:02d}. rate={spec.offered_request_rate:g} "
            f"rep={spec.repetition} requests={options.requests}"
        )
        print("    " + " ".join(command))


def parse_rates(value: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rates must be comma-separated numbers") from exc
    if not rates:
        raise argparse.ArgumentTypeError("at least one rate is required")
    return rates
