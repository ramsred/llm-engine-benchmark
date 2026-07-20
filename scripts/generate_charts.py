#!/usr/bin/env python3
"""Generate decision-oriented SVG charts from an accepted benchmark summary CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
from pathlib import Path


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(title: str, body: str, width: int = 900, height: int = 520) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{esc(title)}</title>
  <rect width="100%" height="100%" fill="#0f172a"/>
  <text x="36" y="42" fill="#f8fafc" font-family="sans-serif" font-size="24" font-weight="700">{esc(title)}</text>
  {body}
</svg>
'''


def frontier(rows: list[dict[str, str]]) -> str:
    data = []
    for row in rows:
        x = number(row, "ttft_p95_s") or number(row, "ttft_p50_s")
        y = number(row, "output_tps_median") or number(row, "output_tps")
        if y is None:
            request_rps = number(row, "request_throughput_rps")
            y = request_rps * 512 if request_rps is not None else None
        if x is not None and y is not None:
            data.append((x, y, f"{row.get('engine')} C{row.get('concurrency')}"))
    if not data:
        return '<text x="36" y="100" fill="#cbd5e1" font-family="sans-serif">No accepted summary rows with TTFT and output throughput.</text>'
    xmax = max(x for x, _, _ in data) * 1.15 or 1
    ymax = max(y for _, y, _ in data) * 1.2 or 1
    body = '<line x1="90" y1="450" x2="850" y2="450" stroke="#94a3b8"/><line x1="90" y1="90" x2="90" y2="450" stroke="#94a3b8"/>'
    body += '<text x="420" y="500" fill="#cbd5e1" font-family="sans-serif">P95 TTFT (s) — lower is better</text><text x="12" y="270" transform="rotate(-90 12 270)" fill="#cbd5e1" font-family="sans-serif">Output tok/s — higher is better</text>'
    colors = {"vllm": "#60a5fa", "sglang": "#34d399", "tensorrt_llm": "#f59e0b"}
    for x, y, label in data:
        px = 90 + (x / xmax) * 760
        py = 450 - (y / ymax) * 360
        engine = label.split()[0]
        color = colors.get(engine, "#f8fafc")
        body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}"/><text x="{px + 9:.1f}" y="{py + 4:.1f}" fill="#e2e8f0" font-family="sans-serif" font-size="12">{esc(label)}</text>'
    return body


def grouped_bars(rows: list[dict[str, str]], metric: str, title: str) -> str:
    values = [(row.get("engine"), row.get("concurrency"), number(row, metric)) for row in rows]
    values = [(e, c, v) for e, c, v in values if v is not None]
    if not values:
        return f'<text x="36" y="100" fill="#cbd5e1" font-family="sans-serif">No accepted summary rows with {esc(metric)}.</text>'
    maximum = max(v for _, _, v in values) or 1
    body = '<line x1="90" y1="450" x2="850" y2="450" stroke="#94a3b8"/>'
    colors = {"vllm": "#60a5fa", "sglang": "#34d399", "tensorrt_llm": "#f59e0b"}
    width = max(10, 700 / len(values) - 4)
    for index, (engine, concurrency, value) in enumerate(values):
        x = 100 + index * (700 / len(values))
        height = value / maximum * 340
        y = 450 - height
        color = colors.get(engine, "#f8fafc")
        body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/><text x="{x + width / 2:.1f}" y="470" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="10">{esc(engine)} C{esc(concurrency)}</text>'
    body += f'<text x="36" y="80" fill="#cbd5e1" font-family="sans-serif">{esc(title)}</text>'
    return body


def write_chart(path: Path, title: str, body: str) -> None:
    path.write_text(svg_document(title, body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/summaries/combined-summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/charts"))
    args = parser.parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_chart(args.output_dir / "measured-ttft-throughput-operating-points.svg", "Measured TTFT–throughput operating points", frontier(rows))
    write_chart(args.output_dir / "cold-warm-benefit.svg", "Cold-versus-warm benefit", grouped_bars(rows, "ttft_p50_s", "TTFT P50 by engine and concurrency"))
    write_chart(args.output_dir / "scaling-efficiency.svg", "Scaling efficiency", grouped_bars(rows, "request_throughput_rps", "Request throughput by engine and concurrency"))
    write_chart(args.output_dir / "prefix-reuse.svg", "Prefix reuse evidence", grouped_bars(rows, "cache_hit_ratio", "Observed cache-hit ratio"))
    dashboard = '<text x="36" y="90" fill="#cbd5e1" font-family="sans-serif" font-size="16">Charts are generated from accepted report summaries; preliminary rows remain labeled.</text>'
    dashboard += '<text x="36" y="130" fill="#cbd5e1" font-family="sans-serif">See methodology and report provenance before interpreting any comparison.</text>'
    write_chart(args.output_dir / "dashboard.svg", "LLM inference benchmark decision dashboard", dashboard)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    provenance = {
        "generated_by": "llm-engine-benchmark/scripts/generate_charts.py",
        "source_summary": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "benchmark_commit": commit,
        "experiment_status": "preliminary" if any(row.get("status") == "preliminary" for row in rows) else "accepted",
        "accepted_repetitions": sorted({int(row["valid_repetitions"]) for row in rows if row.get("valid_repetitions", "").isdigit()}) or [1],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
