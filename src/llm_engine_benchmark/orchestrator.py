from __future__ import annotations

from collections import defaultdict
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import ClientRunOptions, run_benchmark_client, send_warmup_requests
from .environment import capture_environment
from .metrics import write_metrics_diff
from .normalize import encode, fit_variable_segment, load_pinned_tokenizer, preparation_signature
from .report import generate_report
from .server import DockerEngineServer
from .telemetry import TelemetrySession
from .util import (
    BenchmarkError,
    atomic_write_json,
    ensure_dir,
    load_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_jsonl,
)


RUNTIME_WARMUP_SOURCE = (
    "Runtime initialization request. This text is unrelated to every benchmark prompt. "
    "Return the word READY. Numeric nonce: 314159265358979323846."
)


@dataclass(frozen=True)
class RunOptions:
    engines: tuple[str, ...]
    modes: tuple[str, ...]
    concurrencies: tuple[int, ...]
    repetitions: int
    sample_limit: int
    run_order: str
    skip_image_pull: bool = False
    telemetry_enabled: bool = True
    cooldown_seconds: float = 0.0
    resume: bool = False
    overwrite: bool = False
    keep_going: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RunSpec:
    engine: str
    mode: str
    concurrency: int
    repetition: int
    order_index: int


def build_run_plan(options: RunOptions, seed: int) -> list[RunSpec]:
    plan: list[RunSpec] = []
    order_index = 0
    for repetition in range(1, options.repetitions + 1):
        engines = _engine_order(options.engines, options.run_order, repetition, seed)
        for mode in options.modes:
            for concurrency in options.concurrencies:
                for engine in engines:
                    order_index += 1
                    plan.append(
                        RunSpec(
                            engine=engine,
                            mode=mode,
                            concurrency=concurrency,
                            repetition=repetition,
                            order_index=order_index,
                        )
                    )
    return plan

def select_stratified_records(
    records: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Choose a deterministic task-stratified subset independent of file ordering."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record.get("source") or "unknown"), str(record.get("task") or "unknown"))
        buckets[key].append(dict(record))
    for values in buckets.values():
        values.sort(key=lambda item: str(item.get("sample_id") or ""))

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            values = buckets[key]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        depth += 1
    return selected


def run_experiment(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    options: RunOptions,
) -> dict[str, Any]:
    results_dir = ensure_dir(config["paths"]["results_dir"])
    prepared_dir = Path(config["paths"]["data_dir"]) / "prepared"
    records_paths = {
        "cold": prepared_dir / "cold.jsonl",
        "warm_shared": prepared_dir / "warm_shared.jsonl",
        "exact_repeat": prepared_dir / "cold.jsonl",
    }
    warmup_prefix_path = prepared_dir / "warmup_prefixes.jsonl"

    plan = build_run_plan(options, int(config["project"]["seed"]))
    plan_payload = {
        "created_at": utc_now(),
        "options": {
            "engines": list(options.engines),
            "modes": list(options.modes),
            "concurrencies": list(options.concurrencies),
            "repetitions": options.repetitions,
            "sample_limit": options.sample_limit,
            "run_order": options.run_order,
            "dry_run": options.dry_run,
        },
        "runs": [spec.__dict__ for spec in plan],
        "preparation_signature": preparation_signature(config, lock),
    }
    atomic_write_json(results_dir / "run_plan.json", plan_payload)

    if options.dry_run:
        _print_dry_run(config, lock, plan, results_dir)
        return {"dry_run": True, "planned_runs": len(plan), "plan": str(results_dir / "run_plan.json")}

    for mode in options.modes:
        if not records_paths[mode].exists():
            raise BenchmarkError(f"Prepared request file missing: {records_paths[mode]}")

    capture_environment(config, lock, results_dir)
    tokenizer = load_pinned_tokenizer(config, lock)
    runtime_warmup_prompt = _build_runtime_warmup_prompt(tokenizer)
    image_digests = _prepare_images(config, lock, options, results_dir)
    atomic_write_json(results_dir / "environment" / "docker_images.json", image_digests)
    experiment_scope_signature = _experiment_scope_signature(
        config, lock, options, image_digests
    )
    active_experiment = {
        "created_at": utc_now(),
        "experiment_scope_signature": experiment_scope_signature,
        "preparation_signature": preparation_signature(config, lock),
        "image_digests": image_digests,
        "options": plan_payload["options"],
        "runs": plan_payload["runs"],
    }
    atomic_write_json(results_dir / "active_experiment.json", active_experiment)
    plan_payload["experiment_scope_signature"] = experiment_scope_signature
    plan_payload["resolved_image_digests"] = image_digests
    atomic_write_json(results_dir / "run_plan.json", plan_payload)

    failures: list[dict[str, Any]] = []
    completed = 0
    skipped = 0
    abort_error: Exception | None = None
    for plan_index, spec in enumerate(plan):
        run_dir = _run_directory(results_dir, spec)
        records_path = records_paths[spec.mode]
        selected_records = select_stratified_records(
            list(read_jsonl(records_path)), options.sample_limit
        )
        if len(selected_records) != options.sample_limit:
            raise BenchmarkError(
                f"Requested {options.sample_limit} samples but {records_path} contains only "
                f"{len(selected_records)}"
            )
        run_signature = _run_signature(config, lock, spec, selected_records, image_digests[spec.engine])
        action = _prepare_run_directory(
            run_dir,
            signature=run_signature,
            resume=options.resume,
            overwrite=options.overwrite,
        )
        if action == "skip":
            print(
                f"[run] SKIP valid existing {spec.engine}/{spec.mode}/c{spec.concurrency}/"
                f"run_{spec.repetition:02d}"
            )
            skipped += 1
            continue

        print(
            f"[run] {spec.order_index:02d}/{len(plan):02d} {spec.engine} {spec.mode} "
            f"c{spec.concurrency} repetition {spec.repetition}"
        )
        try:
            _execute_one_run(
                config=config,
                lock=lock,
                tokenizer=tokenizer,
                runtime_warmup_prompt=runtime_warmup_prompt,
                spec=spec,
                run_dir=run_dir,
                records_path=records_path,
                selected_records=selected_records,
                warmup_prefix_path=warmup_prefix_path,
                image_digest=image_digests[spec.engine],
                run_signature=run_signature,
                experiment_scope_signature=experiment_scope_signature,
                options=options,
            )
            completed += 1
        except Exception as exc:
            failure = {
                "spec": spec.__dict__,
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            atomic_write_json(run_dir / "run_failure.json", {**failure, "failed_at": utc_now()})
            print(f"[run] FAILED: {failure['error']}")
            if not options.keep_going:
                abort_error = exc
        finally:
            should_continue = abort_error is None
            if (
                should_continue
                and plan_index < len(plan) - 1
                and options.cooldown_seconds > 0
            ):
                _cooldown(run_dir, options.cooldown_seconds)
        if abort_error is not None:
            break

    report_paths: dict[str, str] = {}
    try:
        generated = generate_report(results_dir)
        report_paths = {key: str(path) for key, path in generated.items()}
    except BenchmarkError as exc:
        if completed or skipped:
            print(f"[report] Warning: {exc}")

    summary = {
        "finished_at": utc_now(),
        "planned_runs": len(plan),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "failed_runs": len(failures),
        "failures": failures,
        "reports": report_paths,
    }
    atomic_write_json(results_dir / "experiment_summary.json", summary)
    if failures:
        detail = f" First failure: {type(abort_error).__name__}: {abort_error}" if abort_error else ""
        raise BenchmarkError(
            f"{len(failures)} benchmark run(s) failed; inspect experiment_summary.json.{detail}"
        )
    return summary


def _execute_one_run(
    *,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    tokenizer,
    runtime_warmup_prompt: str,
    spec: RunSpec,
    run_dir: Path,
    records_path: Path,
    selected_records: list[dict[str, Any]],
    warmup_prefix_path: Path,
    image_digest: str,
    run_signature: str,
    experiment_scope_signature: str,
    options: RunOptions,
) -> None:
    ensure_dir(run_dir)
    selected_ids = [str(record["sample_id"]) for record in selected_records]
    selected_groups = sorted(
        {str(record["group_id"]) for record in selected_records if record.get("group_id") is not None}
    )
    atomic_write_json(
        run_dir / "requests.reference.json",
        {
            "prepared_file": str(records_path.resolve()),
            "prepared_file_sha256": sha256_file(records_path),
            "selected_sample_count": len(selected_records),
            "selected_sample_ids": selected_ids,
            "selected_group_ids": selected_groups,
        },
    )
    started_at = utc_now()
    metadata: dict[str, Any] = {
        "run_signature": run_signature,
        "experiment_scope_signature": experiment_scope_signature,
        "preparation_signature": preparation_signature(config, lock),
        "started_at": started_at,
        "engine": spec.engine,
        "cache_mode": spec.mode,
        "concurrency": spec.concurrency,
        "repetition": spec.repetition,
        "order_index": spec.order_index,
        "sample_count": len(selected_records),
        "sample_ids": selected_ids,
        "group_ids": selected_groups,
        "model": lock["model"],
        "image": config["engines"][spec.engine]["image"],
        "image_digest": image_digest,
        "prompt_tokens": int(config["project"]["input_tokens"]),
        "output_tokens": int(config["project"]["output_tokens"]),
        "shared_prefix_tokens": int(config["project"]["shared_prefix_tokens"]),
        "runtime_warmup_tokens": len(encode(tokenizer, runtime_warmup_prompt)),
        "runtime_warmup_sha256": sha256_text(runtime_warmup_prompt),
        "server_fairness_settings": _fairness_settings(config),
        "benchmark_scope": "serving_performance_only_no_answer_quality_scoring",
        "runtime_warmup_limitation": (
            "The unrelated 32-token warm-up does not precompile long-context paths; "
            "first-request long-context compilation is included in measured behavior."
        ),
        "design_compliant_full_run": _is_full_design(config, options),
        "status": "starting",
    }
    atomic_write_json(run_dir / "run_metadata.json", metadata)

    engine_cfg = config["engines"][spec.engine]
    request_extra = dict(engine_cfg.get("request_extra", {}))
    server = DockerEngineServer(
        engine=spec.engine,
        config=config,
        lock=lock,
        run_dir=run_dir,
        skip_image_pull=True,
    )
    server.image_digest = image_digest
    telemetry_cfg = dict(config.get("telemetry", {}))
    telemetry_cfg["enabled"] = bool(options.telemetry_enabled and telemetry_cfg.get("enabled", True))
    telemetry = TelemetrySession(run_dir, telemetry_cfg)

    try:
        server.start()
        metadata["server_command"] = server.run_command
        metadata["status"] = "server_started"
        atomic_write_json(run_dir / "run_metadata.json", metadata)
        server.wait_ready(float(config["project"]["readiness_timeout_seconds"]))
        metadata["runtime_versions"] = server.capture_runtime_versions()
        atomic_write_json(run_dir / "run_metadata.json", metadata)

        warmup_records: list[tuple[str, str]] = [("runtime_warmup", runtime_warmup_prompt)]
        if spec.mode == "warm_shared":
            warmup_by_group = {
                str(record["group_id"]): str(record["prompt"])
                for record in read_jsonl(warmup_prefix_path)
            }
            missing = [group_id for group_id in selected_groups if group_id not in warmup_by_group]
            if missing:
                raise BenchmarkError(f"Missing warm-up prompts for groups: {missing}")
            warmup_records.extend(
                (f"prefix_warmup:{group_id}", warmup_by_group[group_id])
                for group_id in selected_groups
            )
        elif spec.mode == "exact_repeat":
            warmup_records.extend(
                (f"exact_population:{record['sample_id']}", str(record["prompt"]))
                for record in selected_records
            )

        warmup_results = send_warmup_requests(
            base_url=server.base_url,
            model=str(config["project"]["model"]),
            prompts=warmup_records,
            tokenizer=tokenizer,
            request_extra=request_extra,
            timeout_seconds=float(config["project"]["request_timeout_seconds"]),
        )
        atomic_write_json(
            run_dir / "warmup_results.json",
            {
                "cache_mode": spec.mode,
                "completed_at": utc_now(),
                "results": warmup_results,
            },
        )

        server.snapshot_metrics("metrics_before.prom")
        telemetry.start()
        try:
            client_options = ClientRunOptions(
                base_url=server.base_url,
                model=str(config["project"]["model"]),
                engine=spec.engine,
                cache_mode=spec.mode,
                concurrency=spec.concurrency,
                output_tokens=int(config["project"]["output_tokens"]),
                request_timeout_seconds=float(config["project"]["request_timeout_seconds"]),
                request_extra=request_extra,
                save_outputs=bool(config["project"].get("save_outputs", True)),
                require_server_token_usage=bool(
                    config["project"].get("require_server_token_usage", True)
                ),
            )
            result = run_benchmark_client(
                records_path=records_path,
                run_dir=run_dir,
                tokenizer=tokenizer,
                options=client_options,
                sample_limit=len(selected_records),
                sample_ids=selected_ids,
            )
        finally:
            telemetry.stop()
        server.snapshot_metrics("metrics_after.prom")
        write_metrics_diff(
            run_dir / "metrics_before.prom",
            run_dir / "metrics_after.prom",
            run_dir / "metrics_diff.json",
        )

        if spec.mode == "warm_shared":
            unique_tokens = int(config["project"]["input_tokens"]) - int(
                config["project"]["shared_prefix_tokens"]
            )
            measured_wall = float(result.get("measured_wall_time_seconds") or 0.0)
            protocol_estimate = unique_tokens * int(result["successful_requests"])
            result["protocol_unique_suffix_tokens_estimate"] = protocol_estimate
            result["protocol_unique_suffix_throughput_estimate"] = (
                protocol_estimate / measured_wall
                if measured_wall > 0
                else None
            )
            result["shared_prefix_tokens_per_request"] = int(
                config["project"]["shared_prefix_tokens"]
            )
            result["unique_suffix_tokens_per_request"] = unique_tokens
            if result.get("computed_input_tokens_estimate") is None:
                result["computed_input_tokens_estimate"] = protocol_estimate
                result["computed_input_token_count_source"] = (
                    "warm_shared_protocol_lower_bound"
                )
                result["computed_input_throughput_tokens_per_second"] = (
                    protocol_estimate / measured_wall if measured_wall > 0 else None
                )
                result["computed_input_estimate_note"] = (
                    f"Lower-bound protocol estimate: {unique_tokens} unique suffix tokens per "
                    "successful request. Confirm actual cache hits with cache usage details, "
                    "metrics_diff.json, and server.log."
                )
            else:
                observed_note = str(result.get("computed_input_estimate_note") or "").strip()
                result["computed_input_estimate_note"] = (
                    f"{observed_note} The warm protocol lower bound is {protocol_estimate} unique "
                    "suffix tokens across successful requests."
                ).strip()
            atomic_write_json(run_dir / "client_results.json", result)

        server_running_after_measurement = server.is_running()
        metadata["server_running_after_measurement"] = server_running_after_measurement
        if not server_running_after_measurement:
            result["valid"] = False
            validation_errors = list(result.get("validation_errors", []))
            validation_errors.append("engine container was not running after measurement")
            result["validation_errors"] = validation_errors
            atomic_write_json(run_dir / "client_results.json", result)

        metadata["finished_at"] = utc_now()
        metadata["status"] = "accepted" if result["valid"] else "rejected"
        metadata["client_valid"] = bool(result["valid"])
        metadata["validation_errors"] = result.get("validation_errors", [])
        metadata["validation_warnings"] = result.get("validation_warnings", [])
        metadata["generated_output_tokens"] = result.get("generated_output_tokens")
        metadata["expected_generated_output_tokens"] = result.get(
            "expected_generated_output_tokens"
        )
        atomic_write_json(run_dir / "run_metadata.json", metadata)
        if not result["valid"]:
            raise BenchmarkError(
                "Neutral client rejected the run: " + "; ".join(result["validation_errors"])
            )
    finally:
        # stop() is idempotent and captures logs/inspect before removing the container.
        try:
            telemetry.stop()
        except Exception:
            pass
        server.stop()


def _build_runtime_warmup_prompt(tokenizer) -> str:
    prompt, token_ids, _ = fit_variable_segment(
        tokenizer,
        prefix="",
        segment_source=(RUNTIME_WARMUP_SOURCE + " ") * 8,
        suffix=" Return only READY.",
        target_tokens=32,
        label="runtime-warmup",
    )
    if len(token_ids) != 32:
        raise BenchmarkError(
            f"Runtime warm-up has {len(token_ids)} tokens instead of the required 32"
        )
    return prompt


def _prepare_images(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    options: RunOptions,
    results_dir: Path,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for engine in options.engines:
        server = DockerEngineServer(
            engine=engine,
            config=config,
            lock=lock,
            run_dir=results_dir / "environment" / f"image_{engine}",
            skip_image_pull=options.skip_image_pull,
        )
        digests[engine] = server.prepare_image()
    return digests


def _print_dry_run(
    config: Mapping[str, Any], lock: Mapping[str, Any], plan: Sequence[RunSpec], results_dir: Path
) -> None:
    print(f"Dry-run plan: {len(plan)} sequential server configurations")
    for spec in plan:
        run_dir = _run_directory(results_dir, spec)
        server = DockerEngineServer(
            engine=spec.engine,
            config=config,
            lock=lock,
            run_dir=run_dir,
            skip_image_pull=True,
        )
        args = server._build_server_args()  # Render only; no Docker operation.
        command = server._build_docker_command(args)
        print(
            f"{spec.order_index:02d}. {spec.engine} {spec.mode} c{spec.concurrency} "
            f"rep={spec.repetition}\n    {' '.join(command)}"
        )


def _engine_order(
    engines: Sequence[str], run_order: str, repetition: int, seed: int
) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(engines))
    if len(unique) <= 1:
        return unique
    if run_order == "sglang-first":
        return tuple(sorted(unique, key=lambda name: 0 if name == "sglang" else 1))
    if run_order == "vllm-first":
        return tuple(sorted(unique, key=lambda name: 0 if name == "vllm" else 1))
    if run_order == "random":
        values = list(unique)
        random.Random(seed + repetition * 104729).shuffle(values)
        return tuple(values)
    if run_order != "alternate":
        raise BenchmarkError(f"Unknown run order: {run_order}")
    if repetition == 1:
        preferred = ("sglang", "vllm")
    elif repetition == 2:
        preferred = ("vllm", "sglang")
    else:
        values = list(unique)
        random.Random(seed + repetition * 104729).shuffle(values)
        return tuple(values)
    return tuple(name for name in preferred if name in unique)


def _run_directory(results_dir: Path, spec: RunSpec) -> Path:
    return (
        results_dir
        / spec.engine
        / spec.mode
        / f"c{spec.concurrency}"
        / f"run_{spec.repetition:02d}"
    )


def _prepare_run_directory(
    run_dir: Path, *, signature: str, resume: bool, overwrite: bool
) -> str:
    if run_dir.exists():
        metadata_path = run_dir / "run_metadata.json"
        results_path = run_dir / "client_results.json"
        if resume and metadata_path.exists() and results_path.exists():
            metadata = load_json(metadata_path)
            results = load_json(results_path)
            if metadata.get("run_signature") == signature and results.get("valid") is True:
                return "skip"
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise BenchmarkError(
                f"Run directory already exists: {run_dir}. Use --resume or --overwrite."
            )
    ensure_dir(run_dir)
    return "run"


def _run_signature(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    spec: RunSpec,
    records: Sequence[Mapping[str, Any]],
    image_digest: str,
) -> str:
    material = {
        "preparation_signature": preparation_signature(config, lock),
        "require_server_token_usage": bool(config["project"].get("require_server_token_usage", True)),
        "spec": spec.__dict__,
        "sample_ids": [record["sample_id"] for record in records],
        "prompt_hashes": [record.get("metadata", {}).get("prompt_sha256") for record in records],
        "image_digest": image_digest,
        "engine_config": config["engines"][spec.engine],
    }
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def _experiment_scope_signature(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    options: RunOptions,
    image_digests: Mapping[str, str],
) -> str:
    """Identity used to prevent reports from mixing stale experiment matrices."""

    material = {
        "preparation_signature": preparation_signature(config, lock),
        "engines": list(options.engines),
        "modes": list(options.modes),
        "concurrencies": list(options.concurrencies),
        "repetitions": options.repetitions,
        "sample_limit": options.sample_limit,
        "run_order": options.run_order,
        "telemetry_enabled": options.telemetry_enabled,
        "require_server_token_usage": bool(config["project"].get("require_server_token_usage", True)),
        "cooldown_seconds": options.cooldown_seconds,
        "engine_configs": {
            engine: config["engines"][engine] for engine in options.engines
        },
        "image_digests": dict(image_digests),
    }
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def _fairness_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context_length": int(config["project"]["context_length"]),
        "input_tokens": int(config["project"]["input_tokens"]),
        "output_tokens": int(config["project"]["output_tokens"]),
        "kv_cache_dtype": {
            "vllm": config["engines"]["vllm"]["kv_cache_dtype"],
            "sglang": config["engines"]["sglang"]["kv_cache_dtype"],
        },
        "memory_fraction": {
            "vllm": config["engines"]["vllm"]["gpu_memory_utilization"],
            "sglang": config["engines"]["sglang"]["mem_fraction_static"],
        },
        "prefill_budget": {
            "vllm_max_num_batched_tokens": config["engines"]["vllm"][
                "max_num_batched_tokens"
            ],
            "sglang_chunked_prefill_size": config["engines"]["sglang"][
                "chunked_prefill_size"
            ],
            "limitation": "conceptually comparable, not mechanically identical",
        },
        "prefix_caching": {"vllm": "enabled", "sglang": "RadixAttention default"},
        "cuda_graphs": "enabled/default; eager mode not forced",
    }


def _is_full_design(config: Mapping[str, Any], options: RunOptions) -> bool:
    project = config["project"]
    vllm = config["engines"]["vllm"]
    sglang = config["engines"]["sglang"]
    return (
        str(project["model"]) == "openai/gpt-oss-20b"
        and int(project["input_tokens"]) == 120_000
        and int(project["output_tokens"]) == 512
        and int(project["context_length"]) == 131_072
        and int(project["shared_prefix_tokens"]) == 100_000
        and int(project["warm_groups"]) == 10
        and float(vllm["gpu_memory_utilization"]) == 0.85
        and float(sglang["mem_fraction_static"]) == 0.85
        and int(vllm["max_num_batched_tokens"]) == 8_192
        and bool(project.get("require_server_token_usage", True))
        and int(sglang["chunked_prefill_size"]) == 8_192
        and str(vllm["kv_cache_dtype"]).lower() == "fp8_e4m3"
        and str(sglang["kv_cache_dtype"]).lower() == "fp8_e4m3"
        and bool(vllm.get("request_extra", {}).get("ignore_eos"))
        and bool(sglang.get("request_extra", {}).get("ignore_eos"))
        and vllm.get("request_extra", {}).get("add_special_tokens") is False
        and options.sample_limit == 100
        and options.repetitions == 3
        and set(options.concurrencies) == {1, 2, 4}
        and set(options.modes) >= {"cold", "warm_shared"}
        and set(options.engines) == {"sglang", "vllm"}
    )


def _cooldown(run_dir: Path, seconds: float) -> None:
    started = utc_now()
    print(f"[cooldown] Waiting {seconds:g}s before the next engine/configuration")
    time.sleep(seconds)
    atomic_write_json(
        run_dir / "cooldown.json",
        {"started_at": started, "finished_at": utc_now(), "seconds": seconds},
    )
