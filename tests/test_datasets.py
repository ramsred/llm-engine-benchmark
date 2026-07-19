from __future__ import annotations

import copy
import unittest

from llm_engine_benchmark.datasets import _longbench_category, manifest_signature


class DatasetSignatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "project": {"seed": 7},
            "normalization": {"max_source_context_chars": 4_000_000},
        }
        self.lock = {
            "datasets": {
                "infinitebench": {"commit_sha": "a" * 40},
                "longbench_v2": {"commit_sha": "b" * 40},
            },
            "sources": {"ruler": {"commit_sha": "c" * 40}},
        }

    def test_signature_is_stable(self) -> None:
        self.assertEqual(
            manifest_signature(self.config, self.lock),
            manifest_signature(copy.deepcopy(self.config), copy.deepcopy(self.lock)),
        )

    def test_signature_changes_with_seed_or_source_revision(self) -> None:
        seed_config = copy.deepcopy(self.config)
        seed_config["project"]["seed"] = 8
        changed_lock = copy.deepcopy(self.lock)
        changed_lock["datasets"]["infinitebench"]["commit_sha"] = "d" * 40
        original = manifest_signature(self.config, self.lock)
        self.assertNotEqual(original, manifest_signature(seed_config, self.lock))
        self.assertNotEqual(original, manifest_signature(self.config, changed_lock))

    def test_longbench_categories_follow_design_buckets(self) -> None:
        self.assertEqual(
            _longbench_category({"domain": "Single-Document QA"}),
            "single_multi_document_qa",
        )
        self.assertEqual(
            _longbench_category({"sub_domain": "Long In-context Learning"}),
            "long_icl_dialogue",
        )
        self.assertEqual(
            _longbench_category({"sub_domain": "Long-dialogue History Understanding"}),
            "long_icl_dialogue",
        )
        self.assertEqual(
            _longbench_category({"domain": "Code Repository Understanding"}),
            "code_structured",
        )
        self.assertEqual(
            _longbench_category({"domain": "Long Structured Data Understanding"}),
            "code_structured",
        )


if __name__ == "__main__":
    unittest.main()
