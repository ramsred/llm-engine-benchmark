from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_engine_benchmark.capacity_analysis import (
    CapacityEvidence,
    classify_capacity_evidence,
    generate_capacity_analysis,
)


def _row(rate: float, repetition: int, passed: bool) -> CapacityEvidence:
    return CapacityEvidence(
        offered_rps=rate,
        repetition=repetition,
        valid=True,
        sla_pass=passed,
        achieved_rps=rate,
        success_fraction=1.0 if passed else 0.9,
        rejection_fraction=0.0 if passed else 0.1,
        queue_p95_seconds=0.0 if passed else 20.0,
        ttft_p95_seconds=50.0,
        itl_p95_seconds=1.0,
        e2e_p95_seconds=200.0,
        source="test",
    )


class CapacityAnalysisTests(unittest.TestCase):
    def test_bounded_unstable_rate_produces_transition_interval(self) -> None:
        rows = [
            *[_row(0.014, repetition, True) for repetition in range(1, 4)],
            *[_row(0.016, repetition, True) for repetition in range(1, 4)],
            _row(0.018, 1, True),
            _row(0.018, 2, False),
            _row(0.018, 3, False),
            *[_row(0.020, repetition, False) for repetition in range(1, 4)],
        ]
        result = classify_capacity_evidence(rows, 3)
        self.assertEqual(result["decision_status"], "validated_transition_interval")
        self.assertEqual(result["highest_validated_passing_rps"], 0.016)
        self.assertEqual(result["lowest_validated_failing_rps"], 0.020)
        self.assertEqual(result["recommended_operating_rps"], 0.014)
        self.assertEqual(result["unstable_rates"], [0.018])

    def test_non_monotonic_evidence_is_inconclusive(self) -> None:
        rows = [
            *[_row(0.010, repetition, False) for repetition in range(1, 4)],
            *[_row(0.020, repetition, True) for repetition in range(1, 4)],
        ]
        result = classify_capacity_evidence(rows, 3)
        self.assertEqual(result["decision_status"], "inconclusive_non_monotonic")
        self.assertTrue(result["non_monotonic"])
        self.assertIsNone(result["recommended_operating_rps"])

    def test_generate_analysis_writes_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for rate, passed in ((0.01, True), (0.02, False)):
                rate_dir = source / f"rps_{rate}"
                for repetition in range(1, 4):
                    run = rate_dir / f"run_{repetition:02d}"
                    run.mkdir(parents=True)
                    result = {
                        "offered_request_rate": rate,
                        "valid": True,
                        "sla_pass": passed,
                        "achieved_request_throughput_per_second": rate,
                        "success_fraction": 1.0 if passed else 0.9,
                        "rejection_fraction": 0.0 if passed else 0.1,
                        "queue_seconds": {"p95": 0.0 if passed else 20.0},
                        "client_metrics": {
                            "ttft_seconds": {"p95": 50.0},
                            "itl_seconds": {"p95": 1.0},
                            "e2e_seconds": {"p95": 200.0},
                        },
                        "sla": {"targets": {"queue_p95_seconds": 10.0}},
                    }
                    metadata = {
                        "engine": "tensorrt_llm",
                        "cache_mode": "cold",
                        "runtime_state": "steady",
                        "sample_count": 100,
                        "image_digest": "digest",
                        "model": {"repo_id": "model", "commit_sha": "commit"},
                    }
                    (run / "capacity_results.json").write_text(json.dumps(result))
                    (run / "capacity_metadata.json").write_text(json.dumps(metadata))
            outputs = generate_capacity_analysis(
                source_roots=[source], repetitions=3, output_dir=root / "output"
            )
            self.assertTrue(outputs["summary"].exists())
            self.assertTrue(outputs["report"].exists())
            self.assertTrue(outputs["provenance"].exists())
            self.assertNotIn(b"\r\n", outputs["summary"].read_bytes())
            report = outputs["report"].read_text()
            self.assertIn("validated passing load", report)
            self.assertIn("0.010000", report)


if __name__ == "__main__":
    unittest.main()
