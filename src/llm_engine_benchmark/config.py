from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .util import BenchmarkError, deep_merge, expand_env, set_nested


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ENGINES = frozenset(
    {"sglang", "vllm", "tensorrt_llm", "tensorrt_llm_triton"}
)
_PACKAGED_DEFAULT = Path(__file__).with_name("default.yaml")
_PROJECT_DEFAULT = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CONFIG_PATH = _PROJECT_DEFAULT if _PROJECT_DEFAULT.exists() else _PACKAGED_DEFAULT


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise BenchmarkError(f"Configuration root must be a mapping: {path}")
    return loaded


def load_config(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    default = _load_yaml(DEFAULT_CONFIG_PATH)
    if config_path:
        requested = Path(config_path).expanduser().resolve()
        merged = deep_merge(default, _load_yaml(requested))
    else:
        merged = default

    config = copy.deepcopy(expand_env(merged))
    for dotted_key, value in (overrides or {}).items():
        if value is not None:
            set_nested(config, dotted_key, value)

    _resolve_paths(config)
    validate_config(config)
    return config


def _resolve_paths(config: dict[str, Any]) -> None:
    paths = config.setdefault("paths", {})
    for key in (
        "data_dir",
        "results_dir",
        "lock_file",
        "hf_cache_dir",
        "vllm_cache_dir",
        "triton_cache_dir",
        "sglang_cache_dir",
    ):
        raw = paths.get(key)
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        paths[key] = str(path.resolve())


def validate_config(config: Mapping[str, Any]) -> None:
    project = config.get("project", {})
    required_positive = (
        "input_tokens",
        "output_tokens",
        "context_length",
        "shared_prefix_tokens",
        "warm_groups",
        "samples",
        "repetitions",
    )
    for key in required_positive:
        value = int(project.get(key, 0))
        if value <= 0:
            raise BenchmarkError(f"project.{key} must be positive")

    input_tokens = int(project["input_tokens"])
    output_tokens = int(project["output_tokens"])
    context_length = int(project["context_length"])
    shared_prefix_tokens = int(project["shared_prefix_tokens"])
    if input_tokens + output_tokens > context_length:
        raise BenchmarkError(
            "input_tokens + output_tokens must not exceed context_length "
            f"({input_tokens} + {output_tokens} > {context_length})"
        )
    if shared_prefix_tokens >= input_tokens:
        raise BenchmarkError("shared_prefix_tokens must be smaller than input_tokens")

    samples = int(project["samples"])
    if samples > 100:
        raise BenchmarkError(
            "project.samples cannot exceed the 100-record canonical design suite"
        )
    warm_groups = int(project["warm_groups"])
    if warm_groups > 100 or 100 % warm_groups != 0:
        raise BenchmarkError(
            "project.warm_groups must be a divisor of the 100-record canonical suite"
        )

    normalization = config.get("normalization", {})
    instruction_reserve = int(normalization.get("instruction_reserve_tokens", 0))
    if instruction_reserve <= 0:
        raise BenchmarkError("normalization.instruction_reserve_tokens must be positive")
    if instruction_reserve >= input_tokens:
        raise BenchmarkError(
            "normalization.instruction_reserve_tokens must be smaller than project.input_tokens"
        )

    concurrency = project.get("concurrency", [])
    if not isinstance(concurrency, list) or not concurrency:
        raise BenchmarkError("project.concurrency must be a non-empty list")
    if any(int(value) <= 0 for value in concurrency):
        raise BenchmarkError("all concurrency values must be positive")

    engines = set(project.get("engines", []))
    unknown_engines = engines - SUPPORTED_ENGINES
    if unknown_engines or not engines:
        raise BenchmarkError(f"Unknown or empty engines: {sorted(unknown_engines)}")

    modes = set(project.get("cache_modes", []))
    unknown_modes = modes - {"cold", "warm_shared", "exact_repeat"}
    if unknown_modes or not modes:
        raise BenchmarkError(f"Unknown or empty cache modes: {sorted(unknown_modes)}")

    run_order = str(project.get("run_order", "alternate"))
    if run_order not in {"alternate", "sglang-first", "vllm-first", "random"}:
        raise BenchmarkError(f"Unknown project.run_order: {run_order}")


def relevant_preparation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    project = config["project"]
    return {
        "model": project["model"],
        "model_revision": project.get("model_revision"),
        "tokenizer_revision": project.get("tokenizer_revision"),
        "trust_remote_code": bool(project.get("trust_remote_code", False)),
        "seed": int(project["seed"]),
        "input_tokens": int(project["input_tokens"]),
        "output_tokens": int(project["output_tokens"]),
        "shared_prefix_tokens": int(project["shared_prefix_tokens"]),
        "warm_groups": int(project["warm_groups"]),
        "sources": config["sources"],
        "normalization": config["normalization"],
    }
