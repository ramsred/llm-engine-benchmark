from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .util import atomic_write_json, atomic_write_text, ensure_dir, human_bytes, redact_mapping, utc_now


def run_doctor(config: Mapping[str, Any], *, as_json: bool = False) -> tuple[dict[str, Any], bool]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "python",
            sys.version_info >= (3, 10),
            f"{platform.python_implementation()} {platform.python_version()}",
            critical=True,
        )
    )
    checks.append(_command_check("docker", ["docker", "--version"], critical=True))
    checks.append(_command_check("docker_daemon", ["docker", "info"], critical=True, timeout=30))
    checks.append(_command_check("nvidia_smi", ["nvidia-smi"], critical=True, timeout=30))
    checks.append(_command_check("pidstat", ["pidstat", "-V"], critical=False))
    checks.append(_command_check("git", ["git", "--version"], critical=False))

    for engine in ("vllm", "sglang"):
        port = int(config["engines"][engine]["host_port"])
        available, detail = _port_available(port)
        checks.append(_check(f"port_{port}_{engine}", available, detail, critical=True))

    data_path = ensure_dir(config["paths"]["data_dir"])
    usage = shutil.disk_usage(data_path)
    # The source datasets, tokenizer/model cache, prepared prompts, logs, and
    # container layers can require substantial space. 20 GiB is a conservative
    # project-level warning threshold, not a hard design requirement.
    checks.append(
        _check(
            "disk_space",
            usage.free >= 20 * 1024**3,
            f"{human_bytes(usage.free)} free at {data_path}",
            critical=False,
        )
    )

    report = {
        "checked_at": utc_now(),
        "platform": platform.platform(),
        "checks": checks,
    }
    ok = all(check["ok"] for check in checks if check["critical"])
    report["critical_checks_passed"] = ok
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for check in checks:
            marker = "PASS" if check["ok"] else ("FAIL" if check["critical"] else "WARN")
            print(f"[{marker:4}] {check['name']}: {check['detail']}")
        print("Doctor result: " + ("PASS" if ok else "FAIL"))
    return report, ok


def capture_environment(
    config: Mapping[str, Any], lock: Mapping[str, Any], results_dir: str | Path
) -> Path:
    environment_dir = ensure_dir(Path(results_dir) / "environment")
    commands = {
        "gpu.txt": ["nvidia-smi"],
        "uname.txt": ["uname", "-a"],
        "memory.txt": ["free", "-h"],
        "docker_version.txt": ["docker", "version"],
        "docker_info.txt": ["docker", "info"],
        "python.txt": [sys.executable, "--version"],
        "python_packages.txt": [sys.executable, "-m", "pip", "freeze", "--all"],
    }
    for filename, command in commands.items():
        target = environment_dir / filename
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
            atomic_write_text(
                target,
                f"$ {' '.join(command)}\nexit={completed.returncode}\n{completed.stdout}",
            )
        except Exception as exc:
            atomic_write_text(target, f"$ {' '.join(command)}\nerror={type(exc).__name__}: {exc}\n")
    atomic_write_json(environment_dir / "experiment_lock.json", lock)
    atomic_write_json(environment_dir / "resolved_config.json", redact_mapping(config))
    atomic_write_json(
        environment_dir / "environment_metadata.json",
        {
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": os.getcwd(),
            "config_paths": config["paths"],
        },
    )
    return environment_dir


def _command_check(
    name: str, command: list[str], *, critical: bool, timeout: int = 10
) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return _check(name, False, f"{command[0]} not found in PATH", critical=critical)
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        first_line = (completed.stdout or "").strip().splitlines()
        detail = first_line[0] if first_line else f"exit={completed.returncode}"
        return _check(name, completed.returncode == 0, detail, critical=critical)
    except Exception as exc:
        return _check(name, False, f"{type(exc).__name__}: {exc}", critical=critical)


def _port_available(port: int) -> tuple[bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            return False, f"port is in use: {exc}"
    return True, "available"


def _check(name: str, ok: bool, detail: str, *, critical: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, "critical": critical}
