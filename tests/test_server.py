from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_engine_benchmark.server import DockerEngineServer, _redact_docker_inspect
from llm_engine_benchmark.util import BenchmarkError


class ServerArgumentTests(unittest.TestCase):
    def _config(self, root: Path):
        return {
            "project": {
                "model": "openai/gpt-oss-20b",
                "context_length": 131072,
                "gpus": "all",
            },
            "paths": {
                "hf_cache_dir": str(root / "hf"),
                "vllm_cache_dir": str(root / "vllm"),
                "triton_cache_dir": str(root / "triton"),
                "sglang_cache_dir": str(root / "sglang"),
            },
            "engines": {
                "vllm": {
                    "image": "vllm:test",
                    "container_name": "vllm-test",
                    "host_port": 8000,
                    "container_port": 8000,
                    "gpu_memory_utilization": 0.85,
                    "max_num_batched_tokens": 8192,
                    "kv_cache_dtype": "fp8_e4m3",
                    "kv_cache_dtype_skip_layers": "sliding_window",
                    "dtype": "bfloat16",
                    "quantization": "mxfp4",
                    "generation_config": "vllm",
                    "extra_args": [],
                },
                "sglang": {
                    "image": "sglang:test",
                    "container_name": "sglang-test",
                    "host_port": 30000,
                    "container_port": 30000,
                    "mem_fraction_static": 0.85,
                    "chunked_prefill_size": 8192,
                    "kv_cache_dtype": "fp8_e4m3",
                    "dtype": "bfloat16",
                    "quantization": "mxfp4",
                    "extra_args": [],
                },
                "tensorrt_llm": {
                    "image": "trtllm:test",
                    "container_name": "trtllm-test",
                    "host_port": 8001,
                    "container_port": 8000,
                    "backend": "pytorch",
                    "max_batch_size": 4,
                    "max_num_tokens": 8192,
                    "kv_cache_free_gpu_memory_fraction": 0.85,
                    "kv_cache_dtype": "fp8",
                    "dtype": "bfloat16",
                    "quantization": "mxfp4",
                    "extra_args": [],
                },
                "tensorrt_llm_triton": {
                    "image": "triton:test",
                    "container_name": "triton-test",
                    "host_port": 9000,
                    "container_port": 9000,
                    "model_repository": str(root),
                    "served_model_name": "tensorrt_llm_bls",
                    "deployment": None,
                    "extra_args": [],
                },
            },
        }


    def test_vllm_disables_model_generation_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="vllm",
                config=self._config(root),
                lock={
                    "model": {
                        "commit_sha": revision,
                        "tokenizer_commit_sha": revision,
                    }
                },
                run_dir=root / "run",
                skip_image_pull=True,
            )
            args = server._build_server_args()
            self.assertEqual(args[args.index("--generation-config") + 1], "vllm")

    def test_sglang_rejects_separate_tokenizer_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = {
                "model": {
                    "commit_sha": "a" * 40,
                    "tokenizer_commit_sha": "b" * 40,
                }
            }
            server = DockerEngineServer(
                engine="sglang",
                config=self._config(root),
                lock=lock,
                run_dir=root / "run",
                skip_image_pull=True,
            )
            with self.assertRaises(BenchmarkError):
                server._build_server_args()

    def test_docker_run_uses_resolved_immutable_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            config = self._config(root)
            lock = {
                "model": {
                    "commit_sha": revision,
                    "tokenizer_commit_sha": revision,
                }
            }
            server = DockerEngineServer(
                engine="vllm", config=config, lock=lock, run_dir=root / "run"
            )
            server.image_digest = '["example/repo@sha256:repo"]|sha256:immutable'
            command = server._build_docker_command(["python", "-V"])
            self.assertIn("sha256:immutable", command)
            self.assertNotIn(config["engines"]["vllm"]["image"], command)

    def test_container_name_is_stable_for_stale_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="sglang",
                config=self._config(root),
                lock={
                    "model": {
                        "commit_sha": revision,
                        "tokenizer_commit_sha": revision,
                    }
                },
                run_dir=root / "run",
                skip_image_pull=True,
            )
            self.assertEqual(server.container_name, "sglang-test")

    def test_runtime_version_capture_script_is_valid_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="sglang",
                config=self._config(root),
                lock={
                    "model": {
                        "commit_sha": revision,
                        "tokenizer_commit_sha": revision,
                    }
                },
                run_dir=root / "run",
                skip_image_pull=True,
            )
            server.started = True
            server.server_args = ["python3", "-m", "sglang.launch_server"]

            def fake_run(command, **kwargs):
                self.assertEqual(command[:4], ["docker", "exec", "sglang-test", "python3"])
                compile(command[-1], "<engine-runtime-capture>", "exec")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout='noise\n{"engine_version":"test"}\n',
                    stderr="",
                )

            with mock.patch("llm_engine_benchmark.server.subprocess.run", side_effect=fake_run):
                payload = server.capture_runtime_versions()

            self.assertEqual(payload, {"engine_version": "test"})
            saved = json.loads((root / "run" / "engine_runtime.json").read_text())
            self.assertEqual(saved, payload)

    def test_sglang_uses_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            lock = {
                "model": {
                    "commit_sha": revision,
                    "tokenizer_commit_sha": revision,
                }
            }
            server = DockerEngineServer(
                engine="sglang",
                config=self._config(root),
                lock=lock,
                run_dir=root / "run",
                skip_image_pull=True,
            )
            args = server._build_server_args()
            self.assertEqual(args[args.index("--revision") + 1], revision)


    def test_tensorrt_llm_uses_pinned_openai_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="tensorrt_llm",
                config=self._config(root),
                lock={"model": {"commit_sha": revision, "tokenizer_commit_sha": revision}},
                run_dir=root / "run",
                skip_image_pull=True,
            )
            args = server._build_server_args()
            self.assertEqual(args[:3], ["trtllm-serve", "serve", "openai/gpt-oss-20b"])
            self.assertEqual(args[args.index("--hf_revision") + 1], revision)
            self.assertEqual(args[args.index("--served_model_name") + 1], "openai/gpt-oss-20b")
            self.assertIn("--max_seq_len", args)

    def test_triton_uses_distinct_model_name_and_pinned_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            config = self._config(root)
            config["engines"]["tensorrt_llm_triton"]["deployment"] = {
                "model": "openai/gpt-oss-20b",
                "model_revision": revision,
                "tokenizer_revision": revision,
                "context_length": 131072,
                "dtype": "bfloat16",
                "quantization": "mxfp4",
            }
            server = DockerEngineServer(
                engine="tensorrt_llm_triton",
                config=config,
                lock={"model": {"commit_sha": revision, "tokenizer_commit_sha": revision}},
                run_dir=root / "run",
                skip_image_pull=True,
            )
            args = server._build_server_args()
            self.assertEqual(server.api_model, "tensorrt_llm_bls")
            tokenizer = args[args.index("--tokenizer") + 1]
            self.assertTrue(tokenizer.endswith(f"/snapshots/{revision}"))
            command = server._build_docker_command(args)
            self.assertIn(f"{root.resolve()}:/models:ro", command)

    def test_triton_rejects_unverified_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="tensorrt_llm_triton",
                config=self._config(root),
                lock={"model": {"commit_sha": revision, "tokenizer_commit_sha": revision}},
                run_dir=root / "run",
                skip_image_pull=True,
            )
            with self.assertRaisesRegex(BenchmarkError, "deployment metadata"):
                server._build_server_args()

    def test_hf_token_is_forwarded_by_name_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            revision = "a" * 40
            server = DockerEngineServer(
                engine="vllm",
                config=self._config(root),
                lock={"model": {"commit_sha": revision, "tokenizer_commit_sha": revision}},
                run_dir=root / "run",
                skip_image_pull=True,
            )
            with mock.patch.dict("os.environ", {"HF_TOKEN": "secret-value"}):
                command = server._build_docker_command(["python", "-V"])
            self.assertIn("HF_TOKEN", command)
            self.assertNotIn("secret-value", command)
            payload = _redact_docker_inspect(
                [{"Config": {"Env": ["PATH=/bin", "HF_TOKEN=secret-value"]}}]
            )
            self.assertEqual(payload[0]["Config"]["Env"], ["PATH=/bin", "HF_TOKEN=<redacted>"])


if __name__ == "__main__":
    unittest.main()
