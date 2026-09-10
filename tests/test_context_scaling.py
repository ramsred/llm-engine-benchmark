from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from llm_engine_benchmark.config import load_config
from llm_engine_benchmark.context_scaling import (
    ContextCell,
    ContextOptions,
    accept_result,
    assert_idle,
    build_context_plan,
    build_context_records,
    cell_config,
    load_sources,
    observed_concurrency,
    run_context_scaling,
    validate_prompt_set,
    write_context_report,
)
from llm_engine_benchmark.normalize import _build_cold_records, instruction_suffix_with_budget
from llm_engine_benchmark.util import BenchmarkError, load_json, read_jsonl, write_jsonl

MODULE = "llm_engine_benchmark.context_scaling"


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class FakeServer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.base_url = "http://localhost:8001"
        self.api_model = "test-model"
        self.run_command = ["fake-server"]
        self.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def prepare_image(self):
        return "sha256:test-image"

    def wait_ready(self, timeout):
        pass

    def capture_runtime_versions(self):
        return {"test": "1"}

    def snapshot_metrics(self, filename):
        pass

    def is_running(self):
        return self.started

    def validate_profile_artifacts(self):
        pass


def fake_client(**kwargs):
    directory = Path(kwargs["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(kwargs["records_path"]))
    if kwargs.get("sample_ids") is not None:
        by_id = {r["sample_id"]: r for r in records}
        records = [by_id[key] for key in kwargs["sample_ids"]]
    else:
        records = records[: kwargs["sample_limit"]]
    count = len(records)
    result = {
        "valid": True,
        "successful_requests": count,
        "server_reported_prompt_token_coverage_requests": count,
        "server_reported_cached_prompt_tokens_total": 0,
        "cache_report_coverage_fraction": 1.0,
        "output_throughput_tokens_per_second": 10.0,
        "request_throughput_per_second": 0.1,
    }
    for name in ("ttft", "itl", "tpot", "e2e"):
        result[f"{name}_seconds"] = {"p50": 1.0, "p95": 2.0, "mean": 1.5}
    (directory / "client_results.json").write_text(json.dumps(result))
    write_jsonl(
        directory / "request_timings.jsonl",
        [
            {
                "request_start_offset_seconds": i // kwargs["options"].concurrency,
                "request_end_offset_seconds": i // kwargs["options"].concurrency + 1,
                "itl_seconds": [0.1, 0.3],
            }
            for i in range(count)
        ],
    )
    return result


class ContextScalingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["normalization"]["controlled_distractor_min_chars"] = 128
        self.config["normalization"]["extra_source_tokens"] = 128
        self.options = ContextOptions(output_dir=Path("unused"))

    def test_plan_has_eight_cells_and_reverses_between_repetitions(self):
        options = replace(self.options, repetitions=3)
        plan = build_context_plan(self.config, options)
        self.assertEqual(len(plan), 24)

        def pairs(cells):
            return [(c.input_tokens, c.concurrency) for c in cells]

        self.assertEqual(pairs(plan[:8]), list(reversed(pairs(plan[8:16]))))
        self.assertEqual(plan, build_context_plan(self.config, options))
        self.assertEqual(len(set(pairs(plan[:8]))), 8)

    def test_invalid_options_fail_before_execution(self):
        cases = [
            {"lengths": ()},
            {"lengths": (8000, 8000)},
            {"lengths": (-1,)},
            {"lengths": (131072,)},
            {"lengths": (32,)},
            {"concurrencies": (0,)},
            {"concurrencies": (1, 8)},
            {"samples": 2},
            {"samples": 101},
            {"repetitions": 0},
            {"warmup_waves": 0},
            {"cooldown_seconds": -1},
            {"cooldown_seconds": float("nan")},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(BenchmarkError):
                build_context_plan(self.config, replace(self.options, **changes))

    def test_cell_config_leaves_base_unchanged_and_keeps_server_ceiling(self):
        original = copy.deepcopy(self.config)
        derived = cell_config(self.config, 8000)
        self.assertEqual(derived["project"]["input_tokens"], 8000)
        self.assertEqual(derived["project"]["context_length"], 131072)
        self.assertEqual(derived["engines"], original["engines"])
        self.assertEqual(self.config, original)

    def test_dry_run_does_not_create_files_load_sources_or_contact_gpu(self):
        with tempfile.TemporaryDirectory() as temp, patch(f"{MODULE}.load_sources") as sources:
            root = Path(temp) / "new"
            with patch(f"{MODULE}.assert_idle") as idle:
                result = run_context_scaling(
                    self.config, replace(self.options, output_dir=root, dry_run=True)
                )
            self.assertEqual(result["planned_runs"], 8)
            self.assertFalse(root.exists())
            sources.assert_not_called()
            idle.assert_not_called()

    def test_existing_results_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "existing.txt"
            sentinel.write_text("preserve")
            with self.assertRaisesRegex(BenchmarkError, "already exists"):
                run_context_scaling(self.config, replace(self.options, output_dir=Path(temp)))
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_canonical_data_path_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.config["paths"]["data_dir"] = temp
            with self.assertRaisesRegex(BenchmarkError, "canonical data"):
                run_context_scaling(
                    self.config, replace(self.options, output_dir=Path(temp) / "new")
                )

    def test_existing_source_lock_is_required_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            self.config["paths"]["lock_file"] = str(Path(temp) / "absent.json")
            with self.assertRaisesRegex(BenchmarkError, "Existing"):
                load_sources(self.config)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_exact_lengths_and_prefixes(self):
        tokenizer = CharTokenizer()
        records = [
            {"sample_id": "a", "prompt": "a" * 40, "prompt_tokens": 40},
            {"sample_id": "b", "prompt": "b" * 40, "prompt_tokens": 40},
        ]
        self.assertEqual(validate_prompt_set(tokenizer, records, 40)["input_tokens"], 40)
        records[1]["prompt"] = "a" * 32 + "b" * 8
        with self.assertRaisesRegex(BenchmarkError, "prefix"):
            validate_prompt_set(tokenizer, records, 40)
        records[1]["prompt"] = "b" * 39
        with self.assertRaisesRegex(BenchmarkError, "token count"):
            validate_prompt_set(tokenizer, records, 40)

    def test_eight_common_tokens_are_rejected_even_with_distinct_32_token_heads(self):
        records = [
            {"sample_id": "a", "prompt": "header00" + "a" * 32, "prompt_tokens": 40},
            {"sample_id": "b", "prompt": "header00" + "b" * 32, "prompt_tokens": 40},
        ]
        with self.assertRaisesRegex(BenchmarkError, "shared 8-token prefix"):
            validate_prompt_set(CharTokenizer(), records, 40)

    def test_early_prefix_removes_common_header_without_changing_legacy_builder(self):
        tokenizer = CharTokenizer()
        config = cell_config(self.config, 2048)
        sources = [
            {
                "sample_id": key,
                "source": "synthetic",
                "task": "qa",
                "context": "Document.",
                "instruction": "Preserve this task instruction.",
            }
            for key in ("w0-0", "w0-1", "w1-0", "measured-1")
        ]
        original_sources = copy.deepcopy(sources)
        legacy = _build_cold_records(config, tokenizer, sources)
        with self.assertRaisesRegex(BenchmarkError, "shared 8-token prefix"):
            validate_prompt_set(tokenizer, legacy, 2048)
        current = build_context_records(config, tokenizer, sources)
        self.assertEqual(current, build_context_records(config, tokenizer, sources))
        self.assertEqual(
            validate_prompt_set(tokenizer, current, 2048)["unique_prefix_check_tokens"], 8
        )
        self.assertEqual(sources, original_sources)
        self.assertEqual(legacy, _build_cold_records(config, tokenizer, sources))
        for record, source in zip(current, sources, strict=True):
            suffix, _ = instruction_suffix_with_budget(config, tokenizer, source)
            self.assertTrue(record["prompt"].endswith(suffix))
            self.assertEqual(record["metadata"]["context_prompt_format_version"], 2)

    def test_block_cache_model_observes_no_reuse_across_warmups_and_measurement(self):
        prefixes_by_run = {}

        def cache_client(**kwargs):
            result = fake_client(**kwargs)
            directory = Path(kwargs["run_dir"])
            run = next(p for p in (directory, *directory.parents) if p.name == "run_01")
            seen = prefixes_by_run.setdefault(run, set())
            records = list(read_jsonl(kwargs["records_path"]))
            if kwargs.get("sample_ids") is not None:
                by_id = {r["sample_id"]: r for r in records}
                records = [by_id[key] for key in kwargs["sample_ids"]]
            cached = 0
            for record in records:
                head = tuple(CharTokenizer().encode(record["prompt"])[:8])
                cached += 8 if head in seen else 0
                seen.add(head)
            result["server_reported_cached_prompt_tokens_total"] = cached
            return result

        with tempfile.TemporaryDirectory() as temp:
            _, result, _ = self.mocked_run(temp, client=cache_client)
            self.assertEqual(result["accepted_runs"], 2)

    def test_invalid_missing_usage_and_cache_hits_rejected(self):
        valid = {
            "valid": True,
            "successful_requests": 4,
            "server_reported_prompt_token_coverage_requests": 4,
        }
        accept_result(valid, 4)
        for changes in (
            {"valid": False},
            {"successful_requests": 3},
            {"server_reported_prompt_token_coverage_requests": 0},
            {"server_reported_cached_prompt_tokens_total": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(BenchmarkError):
                accept_result({**valid, **changes}, 4)

    def test_busy_host_guard_uses_read_only_commands(self):
        for outputs in (("llmbench-tensorrt-llm",), ("", "12345")):
            completed = [subprocess.CompletedProcess([], 0, text) for text in outputs]
            with patch(f"{MODULE}.subprocess.run", side_effect=completed) as command:
                with self.assertRaisesRegex(BenchmarkError, "not idle"):
                    assert_idle(self.config)
                self.assertTrue(all("rm" not in c.args[0] for c in command.call_args_list))

    def test_busy_port_guard(self):
        with (
            patch(f"{MODULE}.subprocess.run", return_value=Mock(stdout="")),
            patch(f"{MODULE}.socket.socket") as sock,
        ):
            sock.return_value.__enter__.return_value.bind.side_effect = OSError("busy")
            with self.assertRaisesRegex(BenchmarkError, "Port"):
                assert_idle(self.config)

    def test_concurrency_uses_half_open_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timings.jsonl"
            write_jsonl(
                path,
                [
                    {"request_start_offset_seconds": 0, "request_end_offset_seconds": 1},
                    {"request_start_offset_seconds": 1, "request_end_offset_seconds": 2},
                ],
            )
            self.assertEqual(observed_concurrency(path), 1)

    def mocked_run(self, temp, client=fake_client, **changes):
        root = Path(temp) / "experiment"
        sources = [
            {
                "sample_id": f"sample-{i}",
                "source": "ruler",
                "task": "qa",
                "context": "Document.",
                "instruction": "Answer the question.",
            }
            for i in range(100)
        ]
        FakeServer.instances = []
        options = replace(
            self.options,
            output_dir=root,
            lengths=(2048,),
            concurrencies=(1, 4),
            samples=4,
            cooldown_seconds=0,
            **changes,
        )
        original = copy.deepcopy(self.config)
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.load_sources", return_value=({}, sources, {})))
            stack.enter_context(patch(f"{MODULE}.assert_idle"))
            stack.enter_context(patch(f"{MODULE}.capture_environment"))
            stack.enter_context(
                patch(f"{MODULE}.load_pinned_tokenizer", return_value=CharTokenizer())
            )
            stack.enter_context(patch(f"{MODULE}.DockerEngineServer", FakeServer))
            stack.enter_context(patch(f"{MODULE}.TelemetrySession"))
            stack.enter_context(patch(f"{MODULE}.write_metrics_diff"))
            mock_client = stack.enter_context(
                patch(f"{MODULE}.run_benchmark_client", side_effect=client)
            )
            result = run_context_scaling(self.config, options)
        self.assertEqual(original, self.config)
        return root, result, mock_client

    def test_end_to_end_excludes_warmup_and_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "original-results.json"
            sentinel.write_text("untouched")
            root, result, client = self.mocked_run(temp)
            self.assertEqual(result["accepted_runs"], 2)
            self.assertEqual(client.call_count, 6)  # two waves + measurement per cell
            self.assertEqual(sentinel.read_text(), "untouched")
            self.assertNotIn(b"\r\n", (root / "context_summary.csv").read_bytes())
            for c in (1, 4):
                directory = root / f"input_2048/c{c}/run_01"
                metadata = load_json(directory / "context_metadata.json")
                self.assertEqual(metadata["warmup_client_peak_concurrency"], [c, c])
                self.assertEqual(
                    load_json(directory / "client_results.json")["successful_requests"], 4
                )
            self.assertTrue(load_json(root / "context_provenance.json")["artifact_sha256"])
            owned = [
                s.kwargs["config"]["engines"]["tensorrt_llm"]["container_name"]
                for s in FakeServer.instances
                if "image" != Path(s.kwargs["run_dir"]).name
            ]
            self.assertEqual(len(set(owned)), 2)
            self.assertTrue(all(name.startswith("llmbench-context-") for name in owned))
            self.assertTrue(all(not s.started for s in FakeServer.instances))

    def test_warmup_failure_stops_before_measurement_and_retains_status(self):
        with tempfile.TemporaryDirectory() as temp:

            def fail(**kwargs):
                raise BenchmarkError("warmup failed")

            with self.assertRaisesRegex(BenchmarkError, "warmup failed"):
                self.mocked_run(temp, client=fail)
            root = Path(temp) / "experiment"
            status = load_json(root / "context_status.json")
            self.assertEqual(status["accepted_runs"], 0)
            self.assertTrue((root / "context_failure.json").exists())
            self.assertTrue(all(not s.started for s in FakeServer.instances))

    def test_measurement_rejection_marks_client_invalid(self):
        with tempfile.TemporaryDirectory() as temp:

            def fail(**kwargs):
                result = fake_client(**kwargs)
                if "warmup" not in Path(kwargs["run_dir"]).parts:
                    result["server_reported_cached_prompt_tokens_total"] = 32
                return result

            with self.assertRaisesRegex(BenchmarkError, "cache reuse"):
                self.mocked_run(temp, client=fail)
            results = list((Path(temp) / "experiment").glob("input_*/c*/run_*/client_results.json"))
            self.assertEqual(len(results), 1)
            self.assertFalse(load_json(results[0])["valid"])

    def test_insufficient_warmup_concurrency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:

            def sequential(**kwargs):
                result = fake_client(**kwargs)
                if kwargs["options"].concurrency == 4:
                    write_jsonl(
                        Path(kwargs["run_dir"]) / "request_timings.jsonl",
                        [
                            {"request_start_offset_seconds": i, "request_end_offset_seconds": i + 1}
                            for i in range(4)
                        ],
                    )
                return result

            with self.assertRaisesRegex(BenchmarkError, "Warm-up did not exercise"):
                self.mocked_run(temp, client=sequential)
            self.assertTrue(all(not server.started for server in FakeServer.instances))

    def test_profile_validation_failure_is_not_accepted(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                FakeServer, "validate_profile_artifacts", side_effect=BenchmarkError("missing CUDA")
            ),
        ):
            with self.assertRaisesRegex(BenchmarkError, "missing CUDA"):
                self.mocked_run(temp, profile_nsys=True)
            status = load_json(Path(temp) / "experiment/context_status.json")
            self.assertEqual(status["accepted_runs"], 0)

    def test_source_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "canonical").mkdir()
            (root / "lock.json").write_text("{}")
            (root / "canonical/manifest.jsonl").write_text("{}\n")
            (root / "canonical/manifest_metadata.json").write_text("{}")
            self.config["paths"]["data_dir"] = str(root)
            self.config["paths"]["lock_file"] = str(root / "lock.json")
            with (
                patch(f"{MODULE}._validate_lock"),
                patch(f"{MODULE}._validate_lock_against_config"),
                patch(
                    f"{MODULE}._canonical_manifest_reuse_error", return_value="checksum mismatch"
                ),
            ):
                with self.assertRaisesRegex(BenchmarkError, "checksum mismatch"):
                    load_sources(self.config)

    def test_missing_runs_are_not_reported_as_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = write_context_report(root, [ContextCell(8000, 1, 1)], False)
            self.assertEqual(result["accepted_runs"], 0)
            self.assertIn("not_run", (root / "context_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
