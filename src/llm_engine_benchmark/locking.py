from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .util import BenchmarkError, atomic_write_json, ensure_dir, load_json, run_command, utc_now


LOCK_FORMAT_VERSION = 1


def load_or_create_lock(
    config: Mapping[str, Any], *, refresh: bool = False, acquire_sources: bool = True
) -> dict[str, Any]:
    lock_path = Path(config["paths"]["lock_file"])
    if lock_path.exists() and not refresh:
        lock = load_json(lock_path)
        _validate_lock(lock)
        _validate_lock_against_config(lock, config)
    else:
        lock = _resolve_lock(config)
        atomic_write_json(lock_path, lock)

    if acquire_sources:
        _acquire_ruler_source(config, lock)
    return lock


def _validate_lock(lock: Mapping[str, Any]) -> None:
    if int(lock.get("format_version", 0)) != LOCK_FORMAT_VERSION:
        raise BenchmarkError(
            f"Unsupported lock format: {lock.get('format_version')}; use --refresh-lock"
        )
    for key in ("model", "datasets", "sources"):
        if key not in lock:
            raise BenchmarkError(f"Lock file is missing '{key}'; use --refresh-lock")


def _validate_lock_against_config(
    lock: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Reject silent reuse when a revision-bearing config value changed."""

    project = config["project"]
    sources = config["sources"]
    mismatches: list[str] = []

    model_lock = lock.get("model", {})
    expected_model_revision = project.get("model_revision") or "main"
    expected_tokenizer_revision = (
        project.get("tokenizer_revision") or expected_model_revision
    )
    expected_model_values = {
        "repo_id": str(project["model"]),
        "revision_requested": str(expected_model_revision),
        "tokenizer_revision_requested": str(expected_tokenizer_revision),
    }
    for key, expected in expected_model_values.items():
        actual = model_lock.get(key) if isinstance(model_lock, Mapping) else None
        if actual != expected:
            mismatches.append(f"model.{key}: lock={actual!r}, config={expected!r}")

    dataset_locks = lock.get("datasets", {})
    for name in ("infinitebench", "longbench_v2"):
        source = sources[name]
        expected_values = {
            "repo_id_requested": str(source["repo_id"]),
            "revision_requested": str(source.get("revision") or "main"),
        }
        dataset_lock = (
            dataset_locks.get(name, {}) if isinstance(dataset_locks, Mapping) else {}
        )
        for key, expected in expected_values.items():
            actual = dataset_lock.get(key) if isinstance(dataset_lock, Mapping) else None
            if actual != expected:
                mismatches.append(
                    f"datasets.{name}.{key}: lock={actual!r}, config={expected!r}"
                )

    ruler_cfg = sources["ruler"]
    source_locks = lock.get("sources", {})
    ruler_lock = (
        source_locks.get("ruler", {}) if isinstance(source_locks, Mapping) else {}
    )
    expected_ruler_values = {
        "repo_url": str(ruler_cfg["repo_url"]),
        "revision_requested": str(ruler_cfg.get("revision") or "main"),
    }
    for key, expected in expected_ruler_values.items():
        actual = ruler_lock.get(key) if isinstance(ruler_lock, Mapping) else None
        if actual != expected:
            mismatches.append(f"sources.ruler.{key}: lock={actual!r}, config={expected!r}")

    if mismatches:
        detail = "; ".join(mismatches)
        raise BenchmarkError(
            "experiment.lock.json does not match the resolved configuration: "
            f"{detail}. Use --refresh-lock (and normally --force-prepare) to create a "
            "new experiment identity."
        )


def _resolve_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise BenchmarkError(
            "huggingface_hub is required; launch through ./bench so dependencies are installed"
        ) from exc

    token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=token)
    project = config["project"]
    sources = config["sources"]

    model_revision_requested = project.get("model_revision") or "main"
    try:
        model_info = api.model_info(
            project["model"], revision=model_revision_requested, token=token
        )
    except Exception as exc:  # huggingface-hub uses several HTTP exception types
        raise BenchmarkError(
            f"Unable to resolve model revision for {project['model']}: {exc}"
        ) from exc

    tokenizer_revision_requested = project.get("tokenizer_revision") or model_revision_requested
    if tokenizer_revision_requested == model_revision_requested:
        tokenizer_sha = model_info.sha
    else:
        try:
            tokenizer_info = api.model_info(
                project["model"], revision=tokenizer_revision_requested, token=token
            )
            tokenizer_sha = tokenizer_info.sha
        except Exception as exc:
            raise BenchmarkError(
                f"Unable to resolve tokenizer revision for {project['model']}: {exc}"
            ) from exc

    datasets: dict[str, Any] = {}
    for name in ("infinitebench", "longbench_v2"):
        source = sources[name]
        requested = source.get("revision") or "main"
        try:
            info = api.dataset_info(source["repo_id"], revision=requested, token=token)
        except Exception as exc:
            raise BenchmarkError(
                f"Unable to resolve dataset revision for {source['repo_id']}: {exc}"
            ) from exc
        datasets[name] = {
            "repo_id_requested": source["repo_id"],
            "repo_id_resolved": getattr(info, "id", source["repo_id"]),
            "revision_requested": requested,
            "commit_sha": info.sha,
        }

    ruler = sources["ruler"]
    ruler_requested = ruler.get("revision") or "main"
    ruler_sha = _resolve_git_revision(ruler["repo_url"], ruler_requested)

    return {
        "format_version": LOCK_FORMAT_VERSION,
        "created_at": utc_now(),
        "model": {
            "repo_id": project["model"],
            "revision_requested": model_revision_requested,
            "commit_sha": model_info.sha,
            "tokenizer_revision_requested": tokenizer_revision_requested,
            "tokenizer_commit_sha": tokenizer_sha,
        },
        "datasets": datasets,
        "sources": {
            "ruler": {
                "repo_url": ruler["repo_url"],
                "revision_requested": ruler_requested,
                "commit_sha": ruler_sha,
                "generation_mode": "neutral_builtin_task_families",
            }
        },
    }


def _resolve_git_revision(repo_url: str, revision: str) -> str:
    refs = [revision]
    if not revision.startswith("refs/"):
        refs = [f"refs/heads/{revision}", f"refs/tags/{revision}", revision]
    for ref in refs:
        try:
            completed = subprocess.run(
                ["git", "ls-remote", repo_url, ref],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            break
        if completed.returncode == 0 and completed.stdout.strip():
            sha = completed.stdout.split()[0].strip()
            if len(sha) >= 40:
                return sha
    if len(revision) >= 40 and all(char in "0123456789abcdefABCDEF" for char in revision):
        return revision.lower()
    raise BenchmarkError(
        f"Unable to resolve immutable git revision for {repo_url}@{revision}. "
        "Check network access and git, or provide a 40-character commit SHA."
    )


def _acquire_ruler_source(config: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    ruler_cfg = config["sources"]["ruler"]
    if not bool(ruler_cfg.get("fetch_upstream_source", True)):
        return

    destination = Path(config["paths"]["data_dir"]) / "raw" / "sources" / "RULER"
    expected_sha = lock["sources"]["ruler"]["commit_sha"]
    marker = destination / ".llmbench-source.json"
    if marker.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
            if current.get("commit_sha") == expected_sha:
                return
        except json.JSONDecodeError:
            pass

    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination.parent)

    repo_url = ruler_cfg["repo_url"]
    try:
        run_command(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(destination)],
            timeout=600,
        )
        run_command(["git", "checkout", "--detach", expected_sha], cwd=destination, timeout=300)
    except BenchmarkError as git_error:
        if destination.exists():
            shutil.rmtree(destination)
        try:
            _download_github_archive(repo_url, expected_sha, destination)
        except Exception as archive_error:
            raise BenchmarkError(
                "Failed to acquire the pinned NVIDIA RULER source with both git and "
                f"the GitHub archive fallback. git error: {git_error}; "
                f"archive error: {archive_error}"
            ) from archive_error

    marker.write_text(
        json.dumps(
            {
                "repo_url": repo_url,
                "commit_sha": expected_sha,
                "note": (
                    "Upstream source retained for provenance. Prompt generation uses the "
                    "neutral built-in RULER task-family generator to avoid engine templates."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _download_github_archive(repo_url: str, sha: str, destination: Path) -> None:
    normalized = repo_url.removesuffix(".git").rstrip("/")
    if not normalized.startswith("https://github.com/"):
        raise BenchmarkError("Archive fallback currently supports GitHub HTTPS URLs only")
    archive_url = f"{normalized}/archive/{sha}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="llmbench-ruler-") as temp_dir:
        archive_path = Path(temp_dir) / "source.tar.gz"
        try:
            with urllib.request.urlopen(archive_url, timeout=180) as response:
                archive_path.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise BenchmarkError(f"Unable to download {archive_url}: {exc}") from exc
        extract_root = Path(temp_dir) / "extract"
        extract_root.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extract_root)
        children = [child for child in extract_root.iterdir() if child.is_dir()]
        if len(children) != 1:
            raise BenchmarkError("Unexpected GitHub archive layout")
        shutil.move(str(children[0]), str(destination))


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise BenchmarkError("Refusing unsafe archive path")
    archive.extractall(destination)
