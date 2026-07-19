from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .config import DEFAULT_CONFIG_PATH, SUPPORTED_ENGINES, load_config
from .datasets import build_canonical_manifest
from .environment import run_doctor
from .locking import load_or_create_lock
from .normalize import prepare_prompt_files
from .orchestrator import RunOptions, run_experiment
from .report import generate_report
from .util import BenchmarkError, redact_mapping
from .validate import validate_prepared_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Neutral long-context benchmark for SGLang, vLLM, and TensorRT-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check Docker, GPU, ports, and disk space")
    _add_common_config_flags(doctor)
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    prepare = subparsers.add_parser(
        "prepare", help="Pin revisions, download data, create 120K prompts, and validate"
    )
    _add_common_config_flags(prepare)
    _add_preparation_flags(prepare)

    validate = subparsers.add_parser(
        "validate", help="Re-tokenize and validate the immutable prepared prompt files"
    )
    _add_common_config_flags(validate)
    validate.add_argument("--refresh-lock", action="store_true")
    validate.add_argument("--skip-ruler-source", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="Auto-prepare data and run the complete sequential Docker benchmark matrix",
    )
    _add_common_config_flags(run)
    _add_preparation_flags(run)
    run.add_argument(
        "--engines",
        default=None,
        help=(
            "both, vllm, sglang, tensorrt_llm, tensorrt_llm_triton, or a "
            "comma-separated list; defaults to project.engines"
        ),
    )
    run.add_argument(
        "--modes",
        default=None,
        help="Comma-separated: cold,warm_shared,exact_repeat; defaults to project.cache_modes",
    )
    run.add_argument(
        "--concurrency",
        default=None,
        help="Comma-separated maximum client concurrency values; defaults to project.concurrency",
    )
    run.add_argument("--repetitions", type=int, default=None, help="Defaults to project.repetitions")
    run.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Measured requests per configuration; defaults to project.samples; use 5 or 20 for smoke tests",
    )
    run.add_argument(
        "--run-order",
        choices=("alternate", "sglang-first", "vllm-first", "random"),
        default=None,
        help="Defaults to project.run_order",
    )
    run.add_argument("--cooldown-seconds", type=float, default=None)
    run.add_argument("--skip-image-pull", action="store_true")
    run.add_argument("--no-telemetry", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    report = subparsers.add_parser("report", help="Aggregate completed run artifacts")
    _add_common_config_flags(report)

    show_config = subparsers.add_parser("show-config", help="Print the resolved configuration")
    _add_common_config_flags(show_config)

    return parser


def _add_common_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML override file layered on {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-tokens", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--shared-prefix-tokens", type=int, default=None)
    parser.add_argument("--warm-groups", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="Docker --gpus value, for example all")
    parser.add_argument("--vllm-image", default=None)
    parser.add_argument("--sglang-image", default=None)
    parser.add_argument("--tensorrt-llm-image", default=None)
    parser.add_argument("--tensorrt-llm-triton-image", default=None)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument("--sglang-port", type=int, default=None)
    parser.add_argument("--tensorrt-llm-port", type=int, default=None)
    parser.add_argument("--tensorrt-llm-triton-port", type=int, default=None)
    parser.add_argument(
        "--triton-model-repository",
        default=None,
        help="Host path to a prepared Triton TensorRT-LLM model repository",
    )
    parser.add_argument(
        "--triton-served-model-name",
        default=None,
        help="Client-facing Triton model name (for example tensorrt_llm_bls)",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
        help="Set matched vLLM GPU utilization and SGLang static memory fraction",
    )
    parser.add_argument(
        "--prefill-budget",
        type=int,
        default=None,
        help="Set matched-intent vLLM batched-token and SGLang chunked-prefill values",
    )
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=None,
        help="Append a server argument; use --vllm-extra-arg=--flag",
    )
    parser.add_argument(
        "--sglang-extra-arg",
        action="append",
        default=None,
        help="Append a server argument; use --sglang-extra-arg=--flag",
    )
    parser.add_argument(
        "--tensorrt-llm-extra-arg",
        action="append",
        default=None,
        help="Append a trtllm-serve argument; use --tensorrt-llm-extra-arg=--flag",
    )
    parser.add_argument(
        "--tensorrt-llm-triton-extra-arg",
        action="append",
        default=None,
        help="Append a Triton OpenAI frontend argument",
    )


def _add_preparation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--skip-ruler-source",
        action="store_true",
        help="Do not clone/archive the pinned upstream RULER repository (generator remains available)",
    )


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "paths.data_dir": args.data_dir,
        "paths.results_dir": args.results_dir,
        "project.model": args.model,
        "project.model_revision": args.model_revision,
        "project.tokenizer_revision": args.tokenizer_revision,
        "project.seed": args.seed,
        "project.input_tokens": args.input_tokens,
        "project.output_tokens": args.output_tokens,
        "project.context_length": args.context_length,
        "project.shared_prefix_tokens": args.shared_prefix_tokens,
        "project.warm_groups": args.warm_groups,
        "project.gpus": args.gpus,
        "engines.vllm.image": args.vllm_image,
        "engines.sglang.image": args.sglang_image,
        "engines.tensorrt_llm.image": args.tensorrt_llm_image,
        "engines.tensorrt_llm_triton.image": args.tensorrt_llm_triton_image,
        "engines.vllm.host_port": args.vllm_port,
        "engines.sglang.host_port": args.sglang_port,
        "engines.tensorrt_llm.host_port": args.tensorrt_llm_port,
        "engines.tensorrt_llm_triton.host_port": args.tensorrt_llm_triton_port,
        "engines.tensorrt_llm_triton.model_repository": args.triton_model_repository,
        "engines.tensorrt_llm_triton.served_model_name": args.triton_served_model_name,
        "project.trust_remote_code": args.trust_remote_code,
    }
    if args.memory_fraction is not None:
        overrides["engines.vllm.gpu_memory_utilization"] = args.memory_fraction
        overrides["engines.sglang.mem_fraction_static"] = args.memory_fraction
        overrides["engines.tensorrt_llm.kv_cache_free_gpu_memory_fraction"] = args.memory_fraction
    if args.prefill_budget is not None:
        overrides["engines.vllm.max_num_batched_tokens"] = args.prefill_budget
        overrides["engines.sglang.chunked_prefill_size"] = args.prefill_budget
        overrides["engines.tensorrt_llm.max_num_tokens"] = args.prefill_budget
    if args.kv_cache_dtype is not None:
        overrides["engines.vllm.kv_cache_dtype"] = args.kv_cache_dtype
        overrides["engines.sglang.kv_cache_dtype"] = args.kv_cache_dtype
        overrides["engines.tensorrt_llm.kv_cache_dtype"] = args.kv_cache_dtype
    if args.vllm_extra_arg is not None:
        overrides["engines.vllm.extra_args"] = args.vllm_extra_arg
    if args.sglang_extra_arg is not None:
        overrides["engines.sglang.extra_args"] = args.sglang_extra_arg
    if args.tensorrt_llm_extra_arg is not None:
        overrides["engines.tensorrt_llm.extra_args"] = args.tensorrt_llm_extra_arg
    if args.tensorrt_llm_triton_extra_arg is not None:
        overrides["engines.tensorrt_llm_triton.extra_args"] = args.tensorrt_llm_triton_extra_arg
    if getattr(args, "skip_ruler_source", False):
        overrides["sources.ruler.fetch_upstream_source"] = False
    return load_config(args.config, overrides=overrides)


def _prepare(config: Mapping[str, Any], args: argparse.Namespace):
    lock = load_or_create_lock(
        config,
        refresh=bool(getattr(args, "refresh_lock", False)),
        acquire_sources=not bool(getattr(args, "skip_ruler_source", False)),
    )
    force = bool(getattr(args, "force_prepare", False) or getattr(args, "refresh_lock", False))
    manifest = build_canonical_manifest(
        config,
        lock,
        force=force,
    )
    files = prepare_prompt_files(
        config,
        lock,
        manifest,
        force=force,
    )
    validation = validate_prepared_dataset(config, lock)
    return lock, manifest, files, validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "doctor":
            _, ok = run_doctor(config, as_json=args.json)
            return 0 if ok else 2

        if args.command == "show-config":
            print(json.dumps(redact_mapping(config), indent=2, sort_keys=True))
            return 0

        if args.command == "prepare":
            lock, manifest, files, validation = _prepare(config, args)
            print(f"Canonical manifest: {manifest}")
            print(f"Cold prompts:       {files['cold']}")
            print(f"Warm prompts:       {files['warm_shared']}")
            print(f"Validation:         PASS ({validation['cold_prompts']} samples)")
            return 0

        if args.command == "validate":
            lock = load_or_create_lock(
                config,
                refresh=args.refresh_lock,
                acquire_sources=not args.skip_ruler_source,
            )
            validate_prepared_dataset(config, lock)
            return 0

        if args.command == "report":
            paths = generate_report(config["paths"]["results_dir"])
            for label, path in paths.items():
                print(f"{label}: {path}")
            return 0

        if args.command == "run":
            if args.resume and args.overwrite:
                raise BenchmarkError("--resume and --overwrite are mutually exclusive")
            if args.engines is None:
                engines = _validate_engine_sequence(config["project"]["engines"])
            else:
                engines = _parse_engines(args.engines)
            if args.modes is None:
                modes = tuple(str(value) for value in config["project"]["cache_modes"])
            else:
                modes = tuple(_parse_csv(args.modes))
            invalid_modes = set(modes) - {"cold", "warm_shared", "exact_repeat"}
            if invalid_modes or not modes:
                raise BenchmarkError(f"Unknown or empty modes: {sorted(invalid_modes)}")
            if args.concurrency is None:
                concurrencies = tuple(int(value) for value in config["project"]["concurrency"])
            else:
                concurrencies = tuple(int(value) for value in _parse_csv(args.concurrency))
            if not concurrencies or any(value <= 0 for value in concurrencies):
                raise BenchmarkError("Concurrency values must be a non-empty set of positive integers")
            samples = int(
                args.samples if args.samples is not None else config["project"]["samples"]
            )
            repetitions = int(
                args.repetitions
                if args.repetitions is not None
                else config["project"]["repetitions"]
            )
            run_order = str(
                args.run_order
                if args.run_order is not None
                else config["project"].get("run_order", "alternate")
            )
            if not 1 <= samples <= 100:
                raise BenchmarkError("--samples/project.samples must be between 1 and 100")
            if repetitions <= 0:
                raise BenchmarkError("--repetitions/project.repetitions must be positive")
            if run_order not in {"alternate", "sglang-first", "vllm-first", "random"}:
                raise BenchmarkError(f"Unknown run order: {run_order}")

            if args.dry_run:
                lock = load_or_create_lock(
                    config,
                    refresh=args.refresh_lock,
                    acquire_sources=False,
                )
            else:
                lock, _, _, _ = _prepare(config, args)
            cooldown = (
                float(args.cooldown_seconds)
                if args.cooldown_seconds is not None
                else float(config["project"].get("cooldown_seconds", 0))
            )
            options = RunOptions(
                engines=engines,
                modes=tuple(modes),
                concurrencies=concurrencies,
                repetitions=repetitions,
                sample_limit=samples,
                run_order=run_order,
                skip_image_pull=args.skip_image_pull,
                telemetry_enabled=not args.no_telemetry,
                cooldown_seconds=cooldown,
                resume=args.resume,
                overwrite=args.overwrite,
                keep_going=args.keep_going,
                dry_run=args.dry_run,
            )
            summary = run_experiment(config, lock, options=options)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        parser.error(f"Unhandled command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted; active telemetry/container cleanup was requested.", file=sys.stderr)
        return 130
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _validate_engine_sequence(values: Sequence[Any]) -> tuple[str, ...]:
    engines = tuple(str(value).strip().lower() for value in values if str(value).strip())
    invalid = set(engines) - SUPPORTED_ENGINES
    if invalid or not engines:
        raise BenchmarkError(f"Invalid project.engines selection: {list(values)}")
    return tuple(dict.fromkeys(engines))


def _parse_engines(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "both":
        return ("sglang", "vllm")
    engines = tuple(_parse_csv(normalized))
    invalid = set(engines) - SUPPORTED_ENGINES
    if invalid or not engines:
        raise BenchmarkError(f"Invalid engine selection: {value}")
    return tuple(dict.fromkeys(engines))


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
