from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from llm_engine_benchmark.context_analysis import generate_context_analysis
from llm_engine_benchmark.util import BenchmarkError


def write_source(root: Path, cells: list[tuple[int, int, int]], *, marker: str = "same") -> None:
    root.mkdir()
    plan = {
        "source_sha256": {"manifest": "abc"},
        "lock": {"model": "test"},
        "config": {"project": {"output_tokens": 512}, "engines": {"test": {}}},
        "scope": "context_characterization_no_quality_or_production_capacity_claim",
        "context_prompt_format_version": 3,
        "options": {"samples": 20, "warmup_waves": 2, "profile_nsys": False},
    }
    (root / "context_plan.json").write_text(json.dumps(plan))
    (root / "image_identity.json").write_text(json.dumps({"image_digest": "sha256:image"}))
    for tokens, concurrency, repetition in cells:
        run = root / f"input_{tokens}/c{concurrency}/run_{repetition:02d}"
        run.mkdir(parents=True)
        metadata = {
            "status": "accepted",
            "input_tokens": tokens,
            "concurrency": concurrency,
            "repetition": repetition,
            "requests_sha256": f"{marker}-{tokens}",
            "image_digest": "sha256:image",
        }
        result = {
            "valid": True,
            "successful_requests": 20,
            "failed_requests": 0,
            "server_reported_prompt_token_coverage_requests": 20,
            "server_reported_cached_prompt_tokens_total": 0,
            "output_throughput_tokens_per_second": 40.0 / concurrency,
            "request_throughput_per_second": 0.1,
        }
        for name, value in (
            ("ttft", tokens / 1000 * concurrency),
            ("itl", 0.01 * concurrency),
            ("tpot", 0.02 * concurrency),
            ("e2e", tokens / 500 * concurrency),
        ):
            result[f"{name}_seconds"] = {"p50": value / 2, "p95": value, "mean": value}
        (run / "context_metadata.json").write_text(json.dumps(metadata))
        (run / "client_results.json").write_text(json.dumps(result))
        (run / "request_timings.jsonl").write_text(json.dumps({"itl_seconds": [0.1]}) + "\n")


class ContextAnalysisTests(unittest.TestCase):
    def test_consolidates_split_discovery_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            first = base / "first"
            second = base / "second"
            output = base / "output"
            write_source(first, [(8000, 1, 1), (32000, 1, 1), (32000, 4, 1)])
            write_source(second, [(8000, 4, 1)])

            outputs = generate_context_analysis(
                source_roots=[first, second],
                lengths=(8000, 32000),
                concurrencies=(1, 4),
                repetitions=1,
                output_dir=output,
            )

            with outputs["summary"].open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertNotIn(b"\r\n", outputs["summary"].read_bytes())
            report = outputs["report"].read_text()
            self.assertIn("complete_discovery", report)
            self.assertIn("Concurrency effect", report)
            provenance = json.loads(outputs["provenance"].read_text())
            self.assertTrue(provenance["decision"]["complete"])
            self.assertEqual(provenance["decision"]["accepted_cells"], 4)

    def test_missing_cell_is_reported_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            output = Path(temp) / "output"
            write_source(root, [(8000, 1, 1)])
            outputs = generate_context_analysis(
                source_roots=[root],
                lengths=(8000,),
                concurrencies=(1, 4),
                repetitions=1,
                output_dir=output,
            )
            provenance = json.loads(outputs["provenance"].read_text())
            self.assertEqual(provenance["decision"]["status"], "incomplete")
            self.assertEqual(
                provenance["decision"]["missing_cells"],
                [{"input_tokens": 8000, "concurrency": 4, "repetition": 1}],
            )

    def test_duplicate_cell_across_sources_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            write_source(first, [(8000, 1, 1)])
            write_source(second, [(8000, 1, 1)])
            with self.assertRaisesRegex(BenchmarkError, "Duplicate accepted"):
                generate_context_analysis(
                    source_roots=[first, second],
                    lengths=(8000,),
                    concurrencies=(1,),
                    repetitions=1,
                    output_dir=Path(temp) / "output",
                )

    def test_incompatible_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            write_source(first, [(8000, 1, 1)])
            write_source(second, [(8000, 4, 1)])
            plan = json.loads((second / "context_plan.json").read_text())
            plan["options"]["samples"] = 100
            (second / "context_plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(BenchmarkError, "Incompatible"):
                generate_context_analysis(
                    source_roots=[first, second],
                    lengths=(8000,),
                    concurrencies=(1, 4),
                    repetitions=1,
                    output_dir=Path(temp) / "output",
                )

    def test_measured_prompt_difference_between_sources_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            write_source(first, [(8000, 1, 1)], marker="first")
            write_source(second, [(8000, 4, 1)], marker="second")
            with self.assertRaisesRegex(BenchmarkError, "Measured prompts differ"):
                generate_context_analysis(
                    source_roots=[first, second],
                    lengths=(8000,),
                    concurrencies=(1, 4),
                    repetitions=1,
                    output_dir=Path(temp) / "output",
                )


if __name__ == "__main__":
    unittest.main()
