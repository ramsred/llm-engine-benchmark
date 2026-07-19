from __future__ import annotations

import unittest

from llm_engine_benchmark.normalize import (
    controlled_distractor,
    encode,
    find_prefix_guard,
    fit_variable_segment,
    instruction_suffix_with_budget,
)
from llm_engine_benchmark.util import BenchmarkError


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer()

    def test_exact_fit_preserves_suffix(self) -> None:
        prompt, ids, metadata = fit_variable_segment(
            self.tokenizer,
            prefix="PREFIX|",
            segment_source="abcdefghijklmnopqrstuvwxyz" * 20,
            suffix="|QUESTION",
            target_tokens=100,
            label="test",
        )
        self.assertEqual(len(ids), 100)
        self.assertTrue(prompt.startswith("PREFIX|"))
        self.assertTrue(prompt.endswith("|QUESTION"))
        self.assertIn("fit_method", metadata)

    def test_shared_prefix_guard(self) -> None:
        prefix = "shared-prefix"
        prefix_ids = encode(self.tokenizer, prefix)
        guard = find_prefix_guard(self.tokenizer, prefix, prefix_ids, ["<guard>"])
        full_ids = encode(self.tokenizer, prefix + guard + "unique")
        self.assertEqual(full_ids[: len(prefix_ids)], prefix_ids)

    def test_fit_after_shared_prefix(self) -> None:
        prefix = "P" * 50
        prefix_ids = encode(self.tokenizer, prefix)
        prompt, ids, _ = fit_variable_segment(
            self.tokenizer,
            prefix=prefix,
            segment_source="G" + "abcdef" * 30,
            suffix="QUESTION",
            target_tokens=100,
            label="warm",
            preserve_prefix_ids=prefix_ids,
        )
        self.assertEqual(len(ids), 100)
        self.assertEqual(ids[:50], prefix_ids)
        self.assertTrue(prompt.endswith("QUESTION"))

    def test_controlled_distractor_is_varied_and_long_enough(self) -> None:
        text = controlled_distractor(17, 5000, label="unit")
        self.assertGreaterEqual(len(text), 5000)
        self.assertGreater(len(set(text.split())), 20)

    def test_instruction_reserve_is_enforced(self) -> None:
        config = {"normalization": {"instruction_reserve_tokens": 128}}
        suffix, count = instruction_suffix_with_budget(
            config,
            self.tokenizer,
            {"sample_id": "ok", "instruction": "Answer briefly."},
        )
        self.assertEqual(count, len(suffix))

        with self.assertRaises(BenchmarkError):
            instruction_suffix_with_budget(
                {"normalization": {"instruction_reserve_tokens": 20}},
                self.tokenizer,
                {"sample_id": "too-long", "instruction": "A" * 100},
            )


if __name__ == "__main__":
    unittest.main()
