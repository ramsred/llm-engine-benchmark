from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_engine_benchmark.metrics import parse_prometheus, write_metrics_diff
from llm_engine_benchmark.util import percentile, stable_int


class UtilTests(unittest.TestCase):
    def test_percentile_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_stable_int_is_reproducible(self) -> None:
        self.assertEqual(stable_int(7, "a", 3), stable_int(7, "a", 3))
        self.assertNotEqual(stable_int(7, "a", 3), stable_int(8, "a", 3))

    def test_prometheus_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.prom"
            after = root / "after.prom"
            output = root / "diff.json"
            before.write_text("cache_hits_total 2\nrequests_total 4\n", encoding="utf-8")
            after.write_text("cache_hits_total 7\nrequests_total 9\n", encoding="utf-8")
            payload = write_metrics_diff(before, after, output)
            self.assertEqual(payload["all_changed_metrics"]["cache_hits_total"], 5)
            self.assertIn("cache_hits_total", payload["cache_and_prefill_evidence"])


if __name__ == "__main__":
    unittest.main()
