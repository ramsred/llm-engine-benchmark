from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


class BenchmarkError(RuntimeError):
    """User-facing benchmark error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(value, indent=indent, sort_keys=True) + "\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    ensure_dir(target.parent)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise BenchmarkError(f"Expected object at {path}:{line_number}")
            yield value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    # A textual representation is architecture-independent and sufficient for
    # equality/provenance checks.
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def stable_int(seed: int, *parts: object) -> int:
    material = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def command_text(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd) if cwd else None,
            check=check,
            timeout=timeout,
            env=dict(env) if env else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise BenchmarkError(f"Required command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkError(
            f"Command timed out after {timeout}s: {command_text(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise BenchmarkError(
            f"Command failed ({exc.returncode}): {command_text(command)}\n{message}"
        ) from exc


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean_value = statistics.fmean(values)
    if mean_value == 0:
        return None
    return statistics.stdev(values) / mean_value


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def redact_mapping(value: Any) -> Any:
    def is_secret_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        if normalized in {
            "token",
            "hf_token",
            "access_token",
            "auth_token",
            "api_key",
            "apikey",
            "password",
            "secret",
            "authorization",
        }:
            return True
        return normalized.endswith(("_password", "_secret", "_api_key", "_access_token"))

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if is_secret_key(str(key)):
                result[key] = "<redacted>" if item else item
            else:
                result[key] = redact_mapping(item)
        return result
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    return value


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"
