from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .util import atomic_write_json, ensure_dir, utc_now


class TelemetrySession:
    def __init__(self, run_dir: str | Path, config: Mapping[str, Any]) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.config = config
        self.processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
        self.stop_event = threading.Event()
        self.psutil_thread: threading.Thread | None = None
        self.started_at: str | None = None
        self._stopped = False
        self.status: dict[str, Any] = {
            "nvidia_dmon": "disabled",
            "pidstat": "disabled",
            "psutil": "disabled",
        }

    def start(self) -> None:
        if self.started_at is not None and not self._stopped:
            return
        self._stopped = False
        if not bool(self.config.get("enabled", True)):
            return
        self.started_at = utc_now()
        interval = float(self.config.get("interval_seconds", 1.0))

        if bool(self.config.get("nvidia_dmon", True)) and shutil.which("nvidia-smi"):
            output = (self.run_dir / "telemetry.csv").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", str(max(1, int(interval))), "-o", "DT"],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                self.processes.append(("nvidia_dmon", process, output))
                self.status["nvidia_dmon"] = "running"
            except Exception as exc:
                output.close()
                self.status["nvidia_dmon"] = f"failed: {exc}"
        elif bool(self.config.get("nvidia_dmon", True)):
            self.status["nvidia_dmon"] = "unavailable"

        if bool(self.config.get("pidstat", True)) and shutil.which("pidstat"):
            output = (self.run_dir / "cpu_memory.txt").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    ["pidstat", "-rud", "-p", "ALL", str(max(1, int(interval)))],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                self.processes.append(("pidstat", process, output))
                self.status["pidstat"] = "running"
            except Exception as exc:
                output.close()
                self.status["pidstat"] = f"failed: {exc}"
        elif bool(self.config.get("pidstat", True)):
            self.status["pidstat"] = "unavailable"

        if bool(self.config.get("psutil", True)):
            try:
                import psutil  # noqa: F401

                self.psutil_thread = threading.Thread(
                    target=self._psutil_loop,
                    args=(interval,),
                    name="llmbench-psutil-telemetry",
                    daemon=True,
                )
                self.psutil_thread.start()
                self.status["psutil"] = "running"
            except Exception as exc:
                self.status["psutil"] = f"failed: {exc}"

        atomic_write_json(
            self.run_dir / "telemetry_status.json",
            {"started_at": self.started_at, "status": self.status},
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.stop_event.set()
        if self.psutil_thread is not None:
            self.psutil_thread.join(timeout=5)
        for name, process, output in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                    process.wait(timeout=5)
            output.close()
            self.status[name] = f"stopped ({process.returncode})"
        atomic_write_json(
            self.run_dir / "telemetry_status.json",
            {
                "started_at": self.started_at,
                "stopped_at": utc_now(),
                "status": self.status,
            },
        )

    def _psutil_loop(self, interval: float) -> None:
        import psutil

        output_path = self.run_dir / "host_telemetry.jsonl"
        psutil.cpu_percent(interval=None)
        with output_path.open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
                record = {
                    "timestamp": utc_now(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "load_1m": load[0],
                    "load_5m": load[1],
                    "load_15m": load[2],
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "memory_used": memory.used,
                    "memory_percent": memory.percent,
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_percent": swap.percent,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                self.stop_event.wait(interval)

    def __enter__(self) -> "TelemetrySession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
