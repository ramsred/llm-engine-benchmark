from __future__ import annotations

import unittest

from llm_engine_benchmark.cli import _build_dry_run_lock, _parse_engines


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


if __name__ == "__main__":
    unittest.main()
