from __future__ import annotations

import unittest

from llm_engine_benchmark.cli import _build_dry_run_lock, _parse_engines, build_parser


class CliTests(unittest.TestCase):
    def test_all_selects_three_backends_in_default_order(self) -> None:
        self.assertEqual(_parse_engines("all"), ("vllm", "sglang", "tensorrt_llm"))

    def test_dry_run_lock_is_network_independent(self) -> None:
        config = {
            "project": {
                "model": "openai/gpt-oss-20b",
                "model_revision": None,
                "tokenizer_revision": None,
            }
        }
        lock = _build_dry_run_lock(config)
        self.assertEqual(lock["model"]["commit_sha"], "dry-run")
        self.assertEqual(lock["model"]["tokenizer_commit_sha"], "dry-run")

    def test_run_accepts_nsys_profile_flag(self) -> None:
        args = build_parser().parse_args(["run", "--profile-nsys", "--dry-run"])
        self.assertTrue(args.profile_nsys)

    def test_capacity_parses_open_loop_and_admission_options(self) -> None:
        args = build_parser().parse_args(
            [
                "capacity",
                "--rates",
                "0.01,0.02",
                "--arrival-pattern",
                "poisson",
                "--max-in-flight",
                "4",
                "--queue-limit",
                "8",
                "--runtime-state",
                "cold-start",
                "--runtime-warmup-output-tokens",
                "16",
            ]
        )
        self.assertEqual(args.rates, (0.01, 0.02))
        self.assertEqual(args.arrival_pattern, "poisson")
        self.assertEqual(args.max_in_flight, 4)
        self.assertEqual(args.queue_limit, 8)
        self.assertEqual(args.runtime_state, "cold-start")
        self.assertEqual(args.runtime_warmup_output_tokens, 16)


if __name__ == "__main__":
    unittest.main()
