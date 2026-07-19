from __future__ import annotations

import collections
import unittest

from llm_engine_benchmark.ruler import generate_ruler_records


class RulerGeneratorTests(unittest.TestCase):
    def test_all_categories_have_ten_unique_records(self) -> None:
        records = generate_ruler_records(20250214)
        self.assertEqual(len(records), 40)
        self.assertEqual(len({record["sample_id"] for record in records}), 40)
        counts = collections.Counter(record["metadata"]["category"] for record in records)
        self.assertEqual(
            counts,
            {
                "needle/retrieval": 10,
                "variable tracking or multi-hop": 10,
                "aggregation/counting": 10,
                "QA": 10,
            },
        )

    def test_aggregation_answer_matches_actual_context_counts(self) -> None:
        records = generate_ruler_records(20250214)
        for record in records:
            if record["task"] != "aggregation_counting":
                continue
            words = record["context"].split()
            tracked_counts = record["metadata"]["tracked_counts"]
            observed = {word: words.count(word) for word in tracked_counts}
            self.assertEqual(observed, tracked_counts)
            self.assertEqual(record["answer"], max(observed, key=observed.get))


if __name__ == "__main__":
    unittest.main()
