from __future__ import annotations

import copy
import unittest

from llm_engine_benchmark.config import load_config
from llm_engine_benchmark.locking import _validate_lock_against_config
from llm_engine_benchmark.util import BenchmarkError


class LockCompatibilityTests(unittest.TestCase):
    def _lock(self):
        config = load_config()
        lock = {
            "format_version": 1,
            "model": {
                "repo_id": config["project"]["model"],
                "revision_requested": "main",
                "commit_sha": "a" * 40,
                "tokenizer_revision_requested": "main",
                "tokenizer_commit_sha": "a" * 40,
            },
            "datasets": {
                name: {
                    "repo_id_requested": config["sources"][name]["repo_id"],
                    "repo_id_resolved": config["sources"][name]["repo_id"],
                    "revision_requested": "main",
                    "commit_sha": "b" * 40,
                }
                for name in ("infinitebench", "longbench_v2")
            },
            "sources": {
                "ruler": {
                    "repo_url": config["sources"]["ruler"]["repo_url"],
                    "revision_requested": "main",
                    "commit_sha": "c" * 40,
                }
            },
        }
        return config, lock

    def test_matching_lock_is_accepted(self) -> None:
        config, lock = self._lock()
        _validate_lock_against_config(lock, config)

    def test_changed_model_requires_refresh(self) -> None:
        config, lock = self._lock()
        changed = copy.deepcopy(config)
        changed["project"]["model"] = "example/another-model"
        with self.assertRaises(BenchmarkError):
            _validate_lock_against_config(lock, changed)

    def test_changed_dataset_revision_requires_refresh(self) -> None:
        config, lock = self._lock()
        changed = copy.deepcopy(config)
        changed["sources"]["infinitebench"]["revision"] = "v2"
        with self.assertRaises(BenchmarkError):
            _validate_lock_against_config(lock, changed)


if __name__ == "__main__":
    unittest.main()
