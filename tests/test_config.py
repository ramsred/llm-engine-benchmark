from __future__ import annotations

import copy
import unittest

from llm_engine_benchmark.config import validate_config
from llm_engine_benchmark.util import BenchmarkError


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "project": {
                "input_tokens": 120_000,
                "output_tokens": 512,
                "context_length": 131_072,
                "shared_prefix_tokens": 100_000,
                "warm_groups": 10,
                "samples": 100,
                "repetitions": 3,
                "concurrency": [1, 2, 4],
                "engines": ["sglang", "vllm"],
                "cache_modes": ["cold", "warm_shared"],
                "run_order": "alternate",
            },
            "normalization": {"instruction_reserve_tokens": 1024},
        }

    def test_default_shape_is_valid(self) -> None:
        validate_config(self.config)

    def test_instruction_reserve_must_fit_prompt(self) -> None:
        config = copy.deepcopy(self.config)
        config["normalization"]["instruction_reserve_tokens"] = 120_000
        with self.assertRaises(BenchmarkError):
            validate_config(config)

    def test_warm_groups_must_partition_canonical_suite(self) -> None:
        config = copy.deepcopy(self.config)
        config["project"]["warm_groups"] = 6
        with self.assertRaises(BenchmarkError):
            validate_config(config)

    def test_sample_default_can_be_smoke_sized(self) -> None:
        config = copy.deepcopy(self.config)
        config["project"]["samples"] = 5
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
