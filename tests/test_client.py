from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from llm_engine_benchmark.client import (
    ClientRunOptions,
    _aggregate_results,
    run_benchmark_client,
)
from llm_engine_benchmark.util import write_jsonl


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(character) for character in text]


class ClientTests(unittest.TestCase):
    def _options(
        self,
        *,
        output_tokens: int = 4,
        mode: str = "cold",
        require_server_token_usage: bool = True,
    ) -> ClientRunOptions:
        return ClientRunOptions(
            base_url="http://127.0.0.1:1",
            model="test/model",
            engine="vllm",
            cache_mode=mode,
            concurrency=1,
            output_tokens=output_tokens,
            request_timeout_seconds=1,
            request_extra={},
            require_server_token_usage=require_server_token_usage,
        )

    def _success(
        self,
        *,
        server_count: int | None = 4,
        retokenized_count: int = 4,
        prompt_count: int | None = 10,
        cached_prompt_count: int | None = 0,
    ):
        return {
            "sample_id": "sample-1",
            "status": "ok",
            "ttft_seconds": 0.1,
            "e2e_seconds": 1.0,
            "tpot_seconds": 0.3,
            "itl_seconds": [0.2, 0.2, 0.2],
            "input_tokens": 10,
            "expected_output_tokens": 4,
            "actual_output_tokens": server_count if server_count is not None else retokenized_count,
            "server_reported_output_tokens": server_count,
            "retokenized_output_tokens": retokenized_count,
            "server_reported_prompt_tokens": prompt_count,
            "server_reported_cached_prompt_tokens": cached_prompt_count,
            "output_token_count_source": (
                "server_usage" if server_count is not None else "retokenized_text"
            ),
        }

    def test_cold_computed_input_throughput(self) -> None:
        aggregate = _aggregate_results(
            [self._success()], wall_seconds=2.0, options=self._options()
        )
        self.assertTrue(aggregate["valid"])
        self.assertEqual(aggregate["computed_input_tokens_estimate"], 10)
        self.assertEqual(aggregate["computed_input_throughput_tokens_per_second"], 5.0)
        self.assertEqual(
            aggregate["computed_input_token_count_source"],
            "server_usage_cache_details",
        )
        self.assertEqual(aggregate["generated_output_tokens"], 4)

    def test_usage_and_text_count_disagreement_is_diagnostic_warning(self) -> None:
        aggregate = _aggregate_results(
            [self._success(server_count=4, retokenized_count=3)],
            wall_seconds=1.0,
            options=self._options(),
        )
        self.assertTrue(aggregate["valid"])
        self.assertEqual(aggregate["usage_disagreement_sample_ids"], ["sample-1"])
        self.assertEqual(len(aggregate["validation_warnings"]), 1)

    def test_missing_server_output_usage_rejects_strict_run(self) -> None:
        aggregate = _aggregate_results(
            [self._success(server_count=None, retokenized_count=4)],
            wall_seconds=1.0,
            options=self._options(),
        )
        self.assertFalse(aggregate["valid"])
        self.assertEqual(aggregate["missing_server_output_usage_sample_ids"], ["sample-1"])
        self.assertEqual(aggregate["output_count_sources"]["retokenized_text"], 1)
        self.assertTrue(
            any(
                "decoded-text re-tokenization" in warning
                for warning in aggregate["validation_warnings"]
            )
        )
        permissive = _aggregate_results(
            [self._success(server_count=None, retokenized_count=4)],
            wall_seconds=1.0,
            options=self._options(require_server_token_usage=False),
        )
        self.assertTrue(permissive["valid"])



    def test_server_prompt_count_mismatch_rejects_run(self) -> None:
        aggregate = _aggregate_results(
            [self._success(prompt_count=11)],
            wall_seconds=1.0,
            options=self._options(),
        )
        self.assertFalse(aggregate["valid"])
        self.assertEqual(aggregate["prompt_token_mismatch_sample_ids"], ["sample-1"])

    def test_warm_computed_tokens_use_observed_cache_details(self) -> None:
        aggregate = _aggregate_results(
            [self._success(prompt_count=10, cached_prompt_count=8)],
            wall_seconds=2.0,
            options=self._options(mode="warm_shared"),
        )
        self.assertTrue(aggregate["valid"])
        self.assertEqual(aggregate["computed_input_tokens_estimate"], 2)
        self.assertEqual(aggregate["computed_input_throughput_tokens_per_second"], 1.0)
        self.assertEqual(aggregate["cache_hit_ratio"], 0.8)

    def test_run_client_records_input_path_and_usage_count(self) -> None:
        response = {
            "text": "ab",
            "token_ids": [97, 98],
            "token_times_seconds": [0.1, 0.2],
            "first_token_seconds": 0.1,
            "finish_reason": "length",
            "stream_events": 2,
            "request_elapsed_seconds": 0.2,
            "server_reported_completion_tokens": 2,
            "server_reported_prompt_tokens": 6,
            "server_reported_cached_prompt_tokens": 0,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records_path = root / "requests.jsonl"
            write_jsonl(
                records_path,
                [
                    {
                        "sample_id": "sample-1",
                        "source": "ruler",
                        "task": "qa",
                        "prompt": "prompt",
                        "prompt_tokens": 6,
                    }
                ],
            )
            with patch(
                "llm_engine_benchmark.client._stream_completion",
                new=AsyncMock(return_value=response),
            ):
                result = run_benchmark_client(
                    records_path=records_path,
                    run_dir=root / "run",
                    tokenizer=CharTokenizer(),
                    options=self._options(output_tokens=2),
                )
            self.assertTrue(result["valid"])
            self.assertEqual(result["records_path"], str(records_path.resolve()))
            self.assertEqual(result["output_count_sources"]["server_usage"], 1)


if __name__ == "__main__":
    unittest.main()
