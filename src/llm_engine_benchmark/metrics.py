from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .util import atomic_write_json


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+"
    r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)$"
)


def parse_prometheus(path: str | Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    target = Path(path)
    if not target.exists():
        return metrics
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _METRIC_LINE.match(stripped)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        metrics[match.group("name")] = value
    return metrics


def write_metrics_diff(before_path: str | Path, after_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    before = parse_prometheus(before_path)
    after = parse_prometheus(after_path)
    diff: dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if before_value is None:
            continue
        delta = after_value - before_value
        if delta != 0:
            diff[key] = delta
    evidence_keywords = ("cache", "prefix", "hit", "prompt_token", "prefill", "evict")
    evidence = {
        key: value
        for key, value in diff.items()
        if any(keyword in key.lower() for keyword in evidence_keywords)
    }
    payload = {
        "before_metric_count": len(before),
        "after_metric_count": len(after),
        "changed_metric_count": len(diff),
        "cache_and_prefill_evidence": dict(sorted(evidence.items())),
        "all_changed_metrics": dict(sorted(diff.items())),
        "note": (
            "Metric names are engine/version specific. These are raw counter/gauge deltas; "
            "the neutral client does not reinterpret warm logical input throughput as raw prefill speed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload
