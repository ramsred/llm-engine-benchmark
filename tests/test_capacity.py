from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from llm_engine_benchmark.capacity import (
    CapacityClientOptions,
    CapacityOptions,
    CapacitySpec,
    SlaTargets,
    _AdmissionGate,
    _capacity_run_dir,
    _prepare_capacity_run_directory,
    evaluate_sla,
    generate_arrival_offsets,
    generate_capacity_report,
    run_capacity_client,
)
from llm_engine_benchmark.client import ClientRunOptions


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(character) for character in text]


def _sla() -> SlaTargets:
    return SlaTargets(
        ttft_p95_seconds=1.0,
        itl_p95_seconds=0.2,
        e2e_p95_seconds=2.0,
        queue_p95_seconds=0.5,
        min_success_fraction=0.99,
        max_rejection_fraction=0.01,
    )


class CapacityTests(unittest.TestCase):
    def test_constant_and_poisson_arrivals_are_reproducible(self) -> None:
        self.assertEqual(
            generate_arrival_offsets(count=3, rate=2.0, pattern="constant", seed=1),
            [0.0, 0.5, 1.0],
        )
        first = generate_arrival_offsets(count=4, rate=2.0, pattern="poisson", seed=7)
        second = generate_arrival_offsets(count=4, rate=2.0, pattern="poisson", seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_admission_gate_bounds_waiting_requests(self) -> None:
        async def scenario() -> tuple[str, str, int]:
            gate = _AdmissionGate(max_in_flight=1, queue_limit=1)
            first, _ = await gate.acquire(1.0)
            queued = asyncio.create_task(gate.acquire(1.0))
            await asyncio.sleep(0)
            rejected, _ = await gate.acquire(1.0)
            await gate.release()
            admitted, _ = await queued
            await gate.release()
            return first, rejected, gate.maximum_waiting if admitted == "admitted" else -1

        self.assertEqual(
            asyncio.run(scenario()),
            ("admitted", "rejected_queue_full", 1),
        )

    def test_admission_gate_times_out_queued_request(self) -> None:
        async def scenario() -> tuple[str, str]:
            gate = _AdmissionGate(max_in_flight=1, queue_limit=1)
            first, _ = await gate.acquire(1.0)
            timed_out, _ = await gate.acquire(0.001)
            await gate.release()
            return first, timed_out

        self.assertEqual(
            asyncio.run(scenario()),
            ("admitted", "rejected_queue_timeout"),
        )

    def test_resume_skips_valid_run_with_matching_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run_01"
            run_dir.mkdir()
            (run_dir / "capacity_metadata.json").write_text(
                json.dumps(
                    {
                        "run_signature": "same",
                        "experiment_scope_signature": "old-scope",
                    }
                )
            )
            (run_dir / "capacity_results.json").write_text(
                json.dumps({"valid": True})
            )

            action = _prepare_capacity_run_directory(
                run_dir,
                signature="same",
                experiment_scope_signature="new-scope",
                resume=True,
                overwrite=False,
            )

            metadata = json.loads(
                (run_dir / "capacity_metadata.json").read_text()
            )
            self.assertEqual(action, "skip")
            self.assertEqual(metadata["experiment_scope_signature"], "new-scope")
            self.assertEqual(
                metadata["source_experiment_scope_signatures"], ["old-scope"]
            )

    def test_sla_requires_every_check(self) -> None:
        result = {
            "success_fraction": 1.0,
            "rejection_fraction": 0.0,
            "queue_seconds": {"p95": 0.1},
            "client_metrics": {
                "ttft_seconds": {"p95": 0.8},
                "itl_seconds": {"p95": 0.1},
                "e2e_seconds": {"p95": 1.5},
            },
        }
        self.assertTrue(evaluate_sla(result, _sla())["pass"])
        result["rejection_fraction"] = 0.02
        self.assertFalse(evaluate_sla(result, _sla())["pass"])

    def test_open_loop_client_writes_capacity_artifacts(self) -> None:
        fake_result = {
            "request_index": 0,
            "sample_id": "sample",
            "source": "ruler",
            "task": "qa",
            "group_id": None,
            "status": "ok",
            "error": None,
            "queued_seconds": 0.0,
            "request_start_offset_seconds": 0.0,
            "request_end_offset_seconds": 0.01,
            "ttft_seconds": 0.1,
            "e2e_seconds": 1.0,
            "tpot_seconds": 0.1,
            "itl_seconds": [0.1, 0.1, 0.1],
            "itl_p50_seconds": 0.1,
            "itl_p95_seconds": 0.1,
            "input_tokens": 10,
            "expected_output_tokens": 4,
            "actual_output_tokens": 4,
            "output_token_count_source": "server_usage",
            "server_reported_output_tokens": 4,
            "retokenized_output_tokens": 4,
            "server_reported_prompt_tokens": 10,
            "server_reported_cached_prompt_tokens": 0,
            "finish_reason": "length",
            "stream_events": 4,
            "token_event_times_seconds": [0.1, 0.2, 0.3, 0.4],
            "prefix_token_sha256": None,
            "_output_text": "test",
        }
        request_options = ClientRunOptions(
            base_url="http://127.0.0.1:1",
            model="test/model",
            engine="tensorrt_llm",
            cache_mode="cold",
            concurrency=2,
            output_tokens=4,
            request_timeout_seconds=1,
            request_extra={},
        )
        capacity_options = CapacityClientOptions(
            offered_request_rate=1000,
            max_in_flight=2,
            queue_limit=1,
            queue_timeout_seconds=1,
            arrival_pattern="constant",
            seed=1,
            max_arrival_lag_seconds=1,
            sla=_sla(),
        )
        records = [
            {
                "sample_id": f"sample-{index}",
                "source": "ruler",
                "task": "qa",
                "prompt": "prompt",
                "prompt_tokens": 10,
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "llm_engine_benchmark.capacity._bounded_request",
                new=AsyncMock(side_effect=[dict(fake_result), dict(fake_result)]),
            ):
                result = run_capacity_client(
                    records=records,
                    run_dir=temp,
                    tokenizer=CharTokenizer(),
                    request_options=request_options,
                    capacity_options=capacity_options,
                )
            self.assertTrue(result["valid"])
            self.assertTrue(result["sla_pass"])
            self.assertEqual(result["scheduled_requests"], 2)
            self.assertTrue((Path(temp) / "capacity_results.json").exists())
            self.assertTrue((Path(temp) / "capacity_request_timings.jsonl").exists())

    def test_report_selects_highest_rate_passing_every_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            options = CapacityOptions(
                engine="tensorrt_llm",
                mode="cold",
                rates=(0.01, 0.02),
                requests=20,
                repetitions=2,
                max_in_flight=4,
                queue_limit=8,
                queue_timeout_seconds=30,
                arrival_pattern="constant",
                max_arrival_lag_seconds=1,
                sla=_sla(),
                output_dir=root,
            )
            plan = [
                CapacitySpec(rate, repetition, index)
                for index, (rate, repetition) in enumerate(
                    ((0.01, 1), (0.02, 1), (0.01, 2), (0.02, 2)), start=1
                )
            ]
            for spec in plan:
                run_dir = _capacity_run_dir(root, options, spec)
                run_dir.mkdir(parents=True)
                passed = spec.offered_request_rate == 0.01
                payload = {
                    "offered_request_rate": spec.offered_request_rate,
                    "valid": True,
                    "sla_pass": passed,
                    "scheduled_requests": 20,
                    "successful_requests": 20,
                    "rejected_requests": 0,
                    "success_fraction": 1.0,
                    "rejection_fraction": 0.0,
                    "achieved_request_throughput_per_second": 0.01,
                    "actual_offered_request_rate": spec.offered_request_rate,
                    "arrival_lag_seconds": {"p95": 0.01},
                    "queue_seconds": {"p95": 0.0},
                    "client_metrics": {
                        "ttft_seconds": {"p95": 0.8},
                        "itl_seconds": {"p95": 0.1},
                        "e2e_seconds": {"p95": 1.5},
                    },
                }
                (run_dir / "capacity_results.json").write_text(json.dumps(payload))
            report = generate_capacity_report(root, plan, options)
            self.assertEqual(report["capacity_boundary_rps"], 0.01)
            self.assertEqual(report["recommended_safe_capacity_rps"], 0.0075)


if __name__ == "__main__":
    unittest.main()
