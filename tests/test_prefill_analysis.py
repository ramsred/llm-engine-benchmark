from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from llm_engine_benchmark.prefill_analysis import (
    generate_prefill_analysis,
    summarize_budget,
    summarize_profile,
)


class PrefillAnalysisTests(unittest.TestCase):
    def _write_validation(self, root: Path, budget: int) -> None:
        for repetition, value in enumerate((1.0, 2.0, 3.0), start=1):
            run = (
                root
                / f"tensorrt-llm-cold-c4-budget-{budget}"
                / "tensorrt_llm/cold/c4"
                / f"run_{repetition:02d}"
            )
            run.mkdir(parents=True)
            result = {
                "valid": True,
                "successful_requests": 100,
                "ttft_seconds": {"p95": value},
                "itl_seconds": {"p95": value / 10},
                "tpot_seconds": {"mean": value / 100},
                "e2e_seconds": {"p95": value * 2},
                "output_throughput_tokens_per_second": 10 - value,
            }
            (run / "client_results.json").write_text(json.dumps(result), encoding="utf-8")

    def _write_profile(self, root: Path, budget: int) -> Path:
        run = root / f"profile-{budget}"
        (run / "profiling").mkdir(parents=True)
        (run / "client_results.json").write_text(
            json.dumps({"valid": True, "itl_seconds": {"p95": 1.25}}),
            encoding="utf-8",
        )
        connection = sqlite3.connect(run / "profiling/server.sqlite")
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, demangledName INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'cudaEventSynchronize_v3020');
            INSERT INTO StringIds VALUES (2, 'flash_attention_kernel');
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 1500000000, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 125000000, 2);
            """
        )
        connection.commit()
        connection.close()
        return run

    def test_summarizes_repeated_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_validation(root, 2048)
            summary = summarize_budget(root, 2048)
            self.assertEqual(summary.valid_repetitions, 3)
            self.assertEqual(summary.successful_requests, 300)
            self.assertEqual(summary.ttft_p95_seconds.mean, 2.0)
            self.assertEqual(summary.ttft_p95_seconds.sample_stdev, 1.0)

    def test_reads_cuda_profile_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            validation = root / "validation"
            self._write_validation(validation, 2048)
            profile = self._write_profile(root, 2048)
            evidence = summarize_profile(profile, 2048)
            self.assertEqual(evidence.longest_cuda_event_synchronize_seconds, 1.5)
            self.assertEqual(evidence.longest_gpu_kernel_milliseconds, 125.0)

            outputs = generate_prefill_analysis(
                validation_root=validation,
                budgets=(2048,),
                profile_runs={2048: profile},
                output_dir=root / "output",
            )
            self.assertTrue(all(path.exists() for path in outputs.values()))
            report = outputs["report"].read_text(encoding="utf-8")
            self.assertIn("Latency-oriented operating point: **2048 tokens**", report)
            provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
            self.assertEqual(len(provenance["source_sha256"]), 5)


if __name__ == "__main__":
    unittest.main()
