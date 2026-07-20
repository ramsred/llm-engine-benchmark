from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from llm_engine_benchmark.orchestrator import (
    RunOptions,
    _build_runtime_warmup_prompt,
    _prepare_run_directory,
    build_run_plan,
    select_stratified_records,
)


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class OrchestratorTests(unittest.TestCase):
    def test_full_matrix_has_36_runs(self) -> None:
        options = RunOptions(
            engines=("sglang", "vllm"),
            modes=("cold", "warm_shared"),
            concurrencies=(1, 2, 4),
            repetitions=3,
            sample_limit=100,
            run_order="alternate",
        )
        plan = build_run_plan(options, seed=123)
        self.assertEqual(len(plan), 36)
        first_pair = [plan[0].engine, plan[1].engine]
        self.assertEqual(first_pair, ["sglang", "vllm"])
        # Repetition two starts after 12 configurations.
        second_rep_first_pair = [plan[12].engine, plan[13].engine]
        self.assertEqual(second_rep_first_pair, ["vllm", "sglang"])

    def test_alternate_order_rotates_three_backends(self) -> None:
        options = RunOptions(
            engines=("sglang", "vllm", "tensorrt_llm"),
            modes=("cold",),
            concurrencies=(1,),
            repetitions=3,
            sample_limit=1,
            run_order="alternate",
        )
        plan = build_run_plan(options, seed=123)
        self.assertEqual(
            [spec.engine for spec in plan],
            [
                "sglang", "vllm", "tensorrt_llm",
                "vllm", "tensorrt_llm", "sglang",
                "tensorrt_llm", "sglang", "vllm",
            ],
        )

    def test_resume_adopts_valid_run_into_active_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_signature": "same-run",
                        "experiment_scope_signature": "old-scope",
                    }
                )
            )
            (run_dir / "client_results.json").write_text(json.dumps({"valid": True}))

            action = _prepare_run_directory(
                run_dir,
                signature="same-run",
                experiment_scope_signature="combined-scope",
                resume=True,
                overwrite=False,
            )

            self.assertEqual(action, "skip")
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            self.assertEqual(metadata["experiment_scope_signature"], "combined-scope")
            self.assertEqual(metadata["source_experiment_scope_signatures"], ["old-scope"])

    def test_resume_accepts_empty_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            action = _prepare_run_directory(
                run_dir,
                signature="new-run",
                experiment_scope_signature="scope",
                resume=True,
                overwrite=False,
            )
            self.assertEqual(action, "run")

    def test_runtime_warmup_is_exactly_32_tokens(self) -> None:
        prompt = _build_runtime_warmup_prompt(CharTokenizer())
        self.assertEqual(len(CharTokenizer().encode(prompt)), 32)

    def test_partial_selection_is_task_stratified_and_order_independent(self) -> None:
        records = [
            {"sample_id": "a2", "source": "a", "task": "one"},
            {"sample_id": "a1", "source": "a", "task": "one"},
            {"sample_id": "b1", "source": "b", "task": "two"},
            {"sample_id": "c1", "source": "c", "task": "three"},
        ]
        selected = select_stratified_records(records, 3)
        reversed_selected = select_stratified_records(list(reversed(records)), 3)
        self.assertEqual(
            [record["sample_id"] for record in selected],
            [record["sample_id"] for record in reversed_selected],
        )
        self.assertEqual(len({record["task"] for record in selected}), 3)


if __name__ == "__main__":
    unittest.main()
