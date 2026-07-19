from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import SUPPORTED_ENGINES
from .util import (
    BenchmarkError,
    atomic_write_json,
    atomic_write_text,
    command_text,
    ensure_dir,
    run_command,
    utc_now,
)


class DockerEngineServer:
    def __init__(
        self,
        *,
        engine: str,
        config: Mapping[str, Any],
        lock: Mapping[str, Any],
        run_dir: str | Path,
        skip_image_pull: bool = False,
    ) -> None:
        if engine not in SUPPORTED_ENGINES:
            raise BenchmarkError(f"Unsupported engine: {engine}")
        self.engine = engine
        self.config = config
        self.lock = lock
        self.engine_config = config["engines"][engine]
        self.run_dir = ensure_dir(run_dir)
        self.skip_image_pull = skip_image_pull
        base_name = str(self.engine_config.get("container_name", f"llmbench-{engine}"))
        # A stable project-specific name lets a new invocation remove a stale
        # benchmark container left behind by a host reboot or hard interruption.
        self.container_name = base_name
        self.host_port = int(self.engine_config["host_port"])
        self.container_port = int(self.engine_config["container_port"])
        self.image = str(self.engine_config["image"])
        self.container_id: str | None = None
        self.started = False
        self.image_digest: str | None = None
        self.run_command: list[str] = []
        self.server_args: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    @property
    def api_model(self) -> str:
        return str(self.engine_config.get("served_model_name") or self.config["project"]["model"])

    def prepare_image(self) -> str:
        if shutil.which("docker") is None:
            raise BenchmarkError("docker is required but was not found in PATH")
        if not self.skip_image_pull:
            print(f"[server] Pulling image {self.image}")
            completed = subprocess.run(
                ["docker", "pull", self.image],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                local = subprocess.run(
                    ["docker", "image", "inspect", self.image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if local.returncode != 0:
                    raise BenchmarkError(
                        f"Unable to pull Docker image {self.image}: {completed.stderr.strip()}"
                    )
                print(
                    f"[server] Warning: pull failed; using existing local image {self.image}: "
                    f"{completed.stderr.strip()}"
                )
        inspect = run_command(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}|{{.Id}}",
                self.image,
            ]
        )
        self.image_digest = inspect.stdout.strip()
        return self.image_digest

    def start(self) -> None:
        if self.started:
            raise BenchmarkError(f"{self.engine} server is already started")
        self._remove_stale_container()
        self._assert_port_available()
        self._ensure_cache_dirs()
        if self.image_digest is None:
            self.prepare_image()

        self.server_args = self._build_server_args()
        command = self._build_docker_command(self.server_args)
        self.run_command = command
        atomic_write_text(self.run_dir / "server_command.txt", command_text(command) + "\n")
        atomic_write_text(self.run_dir / "image_digest.txt", (self.image_digest or "unknown") + "\n")
        print(f"[server] Starting {self.engine}: {self.container_name}")
        completed = run_command(command, timeout=120)
        self.container_id = completed.stdout.strip()
        if not self.container_id:
            raise BenchmarkError(f"docker run did not return a container ID for {self.engine}")
        self.started = True
        self.capture_inspect()

    def wait_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = self.base_url + "/v1/models"
        last_error = "not attempted"
        print(f"[server] Waiting for {self.engine} readiness at {url}")
        while time.monotonic() < deadline:
            if not self.is_running():
                self.capture_logs()
                tail = _tail_text(self.run_dir / "server.log", 80)
                raise BenchmarkError(
                    f"{self.engine} container exited before readiness. Last logs:\n{tail}"
                )
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read(2_000_000)
                    if 200 <= response.status < 300:
                        try:
                            payload = json.loads(body.decode("utf-8"))
                        except json.JSONDecodeError:
                            payload = None
                        data = payload.get("data") if isinstance(payload, Mapping) else None
                        model_ids = {
                            str(item.get("id"))
                            for item in data or []
                            if isinstance(item, Mapping) and item.get("id") is not None
                        }
                        if self.api_model not in model_ids:
                            last_error = (
                                f"expected model {self.api_model!r} was not advertised; "
                                f"available models: {sorted(model_ids)}"
                            )
                            time.sleep(2)
                            continue
                        atomic_write_json(
                            self.run_dir / "models_readiness.json",
                            {
                                "ready_at": utc_now(),
                                "status": response.status,
                                "payload": payload,
                            },
                        )
                        print(f"[server] {self.engine} is ready")
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = str(exc)
            time.sleep(2)
        self.capture_logs()
        tail = _tail_text(self.run_dir / "server.log", 80)
        raise BenchmarkError(
            f"Timed out after {timeout_seconds}s waiting for {self.engine}. "
            f"Last HTTP error: {last_error}. Last logs:\n{tail}"
        )

    def is_running(self) -> bool:
        if not self.started:
            return False
        completed = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip().lower() == "true"

    def snapshot_metrics(self, filename: str) -> Path:
        target = self.run_dir / filename
        url = self.base_url + "/metrics"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                body = response.read()
                target.write_bytes(body)
        except Exception as exc:
            target.write_text(f"# metrics unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return target

    def capture_inspect(self) -> None:
        if not self.started:
            return
        completed = subprocess.run(
            ["docker", "inspect", self.container_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout)
                atomic_write_json(self.run_dir / "docker_inspect.json", _redact_docker_inspect(payload))
            except json.JSONDecodeError:
                atomic_write_text(self.run_dir / "docker_inspect.txt", completed.stdout)
        else:
            atomic_write_text(self.run_dir / "docker_inspect_error.txt", completed.stderr)

    def capture_runtime_versions(self) -> dict[str, Any] | None:
        if not self.started:
            return None
        module_name = {
            "vllm": "vllm",
            "sglang": "sglang",
            "tensorrt_llm": "tensorrt_llm",
            "tensorrt_llm_triton": "tritonserver",
        }[self.engine]
        script = (
            "import importlib\n"
            "import json\n"
            "import platform\n"
            f"module = importlib.import_module({module_name!r})\n"
            "payload = {\n"
            "    'engine_module': module.__name__,\n"
            "    'engine_version': getattr(module, '__version__', 'unknown'),\n"
            "    'python': platform.python_version(),\n"
            "}\n"
            "try:\n"
            "    import torch\n"
            "    payload['torch_version'] = torch.__version__\n"
            "    payload['torch_cuda_version'] = torch.version.cuda\n"
            "    payload['cuda_available'] = torch.cuda.is_available()\n"
            "except Exception as exc:\n"
            "    payload['torch_error'] = type(exc).__name__ + ': ' + str(exc)\n"
            "print(json.dumps(payload, sort_keys=True))\n"
        )
        executable = "python3"
        completed = subprocess.run(
            ["docker", "exec", self.container_name, executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        target = self.run_dir / "engine_runtime.json"
        if completed.returncode == 0:
            try:
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                payload = json.loads(lines[-1])
                atomic_write_json(target, payload)
                return payload
            except json.JSONDecodeError:
                pass
        atomic_write_text(
            self.run_dir / "engine_runtime_error.txt",
            f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        return None

    def capture_logs(self) -> None:
        target = self.run_dir / "server.log"
        with target.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                ["docker", "logs", "--timestamps", self.container_name],
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                handle.write(f"\n[docker logs failed with status {completed.returncode}]\n")

    def stop(self) -> None:
        if not self.started:
            self._remove_stale_container()
            return
        self.capture_inspect()
        self.capture_logs()
        subprocess.run(
            ["docker", "stop", "--time", "30", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.started = False

    def _assert_port_available(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self.host_port))
            except OSError as exc:
                raise BenchmarkError(
                    f"Host port {self.host_port} is already in use for {self.engine}"
                ) from exc

    def _remove_stale_container(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _ensure_cache_dirs(self) -> None:
        for key in ("hf_cache_dir", "vllm_cache_dir", "triton_cache_dir", "sglang_cache_dir"):
            ensure_dir(self.config["paths"][key])

    def _build_docker_command(self, server_args: Sequence[str]) -> list[str]:
        paths = self.config["paths"]
        project = self.config["project"]
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--gpus",
            str(project.get("gpus", "all")),
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--shm-size",
            "32g",
            "-p",
            f"{self.host_port}:{self.container_port}",
            "-v",
            f"{paths['hf_cache_dir']}:/root/.cache/huggingface",
        ]
        if os.getenv("HF_TOKEN"):
            # Pass by name so the rendered command never contains the credential.
            command.extend(["-e", "HF_TOKEN"])
        if self.engine == "vllm":
            command.extend(
                [
                    "-v",
                    f"{paths['vllm_cache_dir']}:/root/.cache/vllm",
                    "-v",
                    f"{paths['triton_cache_dir']}:/root/.triton_cache",
                    "-e",
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
                    "-e",
                    "TRITON_CACHE_DIR=/root/.triton_cache",
                ]
            )
        elif self.engine == "sglang":
            command.extend(
                [
                    "-v",
                    f"{paths['sglang_cache_dir']}:/root/.cache/sglang",
                ]
            )
        elif self.engine == "tensorrt_llm_triton":
            command.extend(
                [
                    "-v",
                    f"{self._triton_model_repository()}:/models:ro",
                    "-e",
                    "TRTLLM_ORCHESTRATOR=1",
                ]
            )
        command.append(self._runtime_image_reference())
        command.extend(server_args)
        return command

    def _triton_model_repository(self) -> Path:
        raw = self.engine_config.get("model_repository")
        if not raw:
            raise BenchmarkError(
                "tensorrt_llm_triton requires engines.tensorrt_llm_triton."
                "model_repository (or --triton-model-repository)"
            )
        repository = Path(str(raw)).expanduser().resolve()
        if not repository.is_dir():
            raise BenchmarkError(f"Triton model repository does not exist: {repository}")
        return repository

    def _validate_triton_deployment(self) -> None:
        cfg = self.engine_config
        if not cfg.get("served_model_name"):
            raise BenchmarkError(
                "tensorrt_llm_triton requires a non-empty served_model_name"
            )
        deployment = cfg.get("deployment")
        required = (
            "model",
            "model_revision",
            "tokenizer_revision",
            "context_length",
            "dtype",
            "quantization",
        )
        if not isinstance(deployment, Mapping):
            raise BenchmarkError(
                "tensorrt_llm_triton requires deployment metadata declaring the "
                "prebuilt model, revisions, context length, dtype, and quantization"
            )
        missing = [key for key in required if deployment.get(key) is None]
        if missing:
            raise BenchmarkError(
                "Incomplete tensorrt_llm_triton deployment metadata; missing: "
                + ", ".join(missing)
            )
        direct_cfg = self.config["engines"]["tensorrt_llm"]
        expected = {
            "model": str(self.config["project"]["model"]),
            "model_revision": str(self.lock["model"]["commit_sha"]),
            "tokenizer_revision": str(self.lock["model"]["tokenizer_commit_sha"]),
            "context_length": int(self.config["project"]["context_length"]),
            "dtype": str(direct_cfg["dtype"]),
            "quantization": str(direct_cfg["quantization"]),
        }
        mismatches = [
            f"{key}={deployment[key]!r} (expected {value!r})"
            for key, value in expected.items()
            if deployment[key] != value
        ]
        if mismatches:
            raise BenchmarkError(
                "Triton deployment does not match the pinned benchmark configuration: "
                + "; ".join(mismatches)
            )

    def _runtime_image_reference(self) -> str:
        """Run the exact image ID resolved before the experiment, not a mutable tag."""
        if self.image_digest:
            _, separator, image_id = self.image_digest.rpartition("|")
            if separator and image_id.startswith("sha256:"):
                return image_id
        return self.image

    def _build_server_args(self) -> list[str]:
        project = self.config["project"]
        model = str(project["model"])
        revision = str(self.lock["model"]["commit_sha"])
        context_length = str(project["context_length"])
        if self.engine == "vllm":
            cfg = self.engine_config
            args = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--served-model-name",
                model,
                "--revision",
                revision,
                "--tokenizer-revision",
                str(self.lock["model"]["tokenizer_commit_sha"]),
                "--host",
                "0.0.0.0",
                "--port",
                str(self.container_port),
                "--max-model-len",
                context_length,
                "--dtype",
                str(cfg.get("dtype", "bfloat16")),
                "--gpu-memory-utilization",
                str(cfg["gpu_memory_utilization"]),
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                str(cfg["max_num_batched_tokens"]),
                "--kv-cache-dtype",
                str(cfg["kv_cache_dtype"]),
                "--quantization",
                str(cfg.get("quantization", "mxfp4")),
                "--generation-config",
                str(cfg.get("generation_config", "vllm")),
            ]
            skip_layers = cfg.get("kv_cache_dtype_skip_layers")
            if skip_layers:
                args.extend(["--kv-cache-dtype-skip-layers", str(skip_layers)])
            args.extend(str(item) for item in cfg.get("extra_args", []))
            return args

        if self.engine == "tensorrt_llm":
            cfg = self.engine_config
            tokenizer_revision = str(self.lock["model"]["tokenizer_commit_sha"])
            if tokenizer_revision != revision:
                raise BenchmarkError(
                    "trtllm-serve exposes one --hf_revision for this model/tokenizer pair. "
                    "Use the same pinned model and tokenizer commit."
                )
            kv_cache_dtype = str(cfg["kv_cache_dtype"]).lower()
            if kv_cache_dtype not in {"auto", "fp8", "nvfp4"}:
                raise BenchmarkError(
                    "Unsupported TensorRT-LLM KV cache dtype: "
                    f"{kv_cache_dtype}; expected auto, fp8, or nvfp4"
                )
            args = [
                "trtllm-serve",
                "serve",
                model,
                "--tokenizer",
                model,
                "--served_model_name",
                model,
                "--hf_revision",
                revision,
                "--backend",
                str(cfg.get("backend", "pytorch")),
                "--host",
                "0.0.0.0",
                "--port",
                str(self.container_port),
                "--max_seq_len",
                context_length,
                "--max_batch_size",
                str(cfg["max_batch_size"]),
                "--max_num_tokens",
                str(cfg["max_num_tokens"]),
                "--kv_cache_free_gpu_memory_fraction",
                str(cfg["kv_cache_free_gpu_memory_fraction"]),
                "--kv_cache_dtype",
                kv_cache_dtype,
                "--no-telemetry",
            ]
            if bool(project.get("trust_remote_code", False)):
                args.append("--trust_remote_code")
            args.extend(str(item) for item in cfg.get("extra_args", []))
            return args

        if self.engine == "tensorrt_llm_triton":
            cfg = self.engine_config
            self._validate_triton_deployment()
            tokenizer_revision = str(self.lock["model"]["tokenizer_commit_sha"])
            cache_name = model.replace("/", "--")
            tokenizer = (
                f"/root/.cache/huggingface/hub/models--{cache_name}/snapshots/"
                f"{tokenizer_revision}"
            )
            args = [
                "python3",
                "/opt/tritonserver/python/openai/openai_frontend/main.py",
                "--model-repository",
                "/models",
                "--tokenizer",
                tokenizer,
                "--host",
                "0.0.0.0",
                "--port",
                str(self.container_port),
            ]
            args.extend(str(item) for item in cfg.get("extra_args", []))
            return args

        cfg = self.engine_config
        tokenizer_revision = str(self.lock["model"]["tokenizer_commit_sha"])
        if tokenizer_revision != revision:
            raise BenchmarkError(
                "SGLang exposes one --revision for both model and tokenizer in this harness. "
                "For a fair run, use the same pinned model and tokenizer commit."
            )
        args = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--served-model-name",
            model,
            "--revision",
            revision,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.container_port),
            "--context-length",
            context_length,
            "--dtype",
            str(cfg.get("dtype", "bfloat16")),
            "--kv-cache-dtype",
            str(cfg["kv_cache_dtype"]),
            "--chunked-prefill-size",
            str(cfg["chunked_prefill_size"]),
            "--mem-fraction-static",
            str(cfg["mem_fraction_static"]),
            "--quantization",
            str(cfg.get("quantization", "mxfp4")),
        ]
        args.extend(str(item) for item in cfg.get("extra_args", []))
        return args

    def __enter__(self) -> "DockerEngineServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

def _redact_docker_inspect(payload: Any) -> Any:
    """Remove credentials Docker expands into Config.Env before saving evidence."""

    if not isinstance(payload, list):
        return payload
    redacted = json.loads(json.dumps(payload))
    for item in redacted:
        if not isinstance(item, dict):
            continue
        config = item.get("Config")
        if not isinstance(config, dict) or not isinstance(config.get("Env"), list):
            continue
        clean_env = []
        for entry in config["Env"]:
            key = str(entry).split("=", 1)[0].upper()
            clean_env.append(f"{key}=<redacted>" if key == "HF_TOKEN" else entry)
        config["Env"] = clean_env
    return redacted



def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return "<no log file>"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])
