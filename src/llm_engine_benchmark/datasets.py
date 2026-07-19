from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ruler import RULER_GENERATOR_VERSION, generate_ruler_records
from .util import (
    BenchmarkError,
    atomic_write_json,
    ensure_dir,
    load_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    stable_int,
    utc_now,
    write_jsonl,
)


MANIFEST_FORMAT_VERSION = 1
EXPECTED_SOURCE_ALLOCATION = {"ruler": 40, "infinitebench": 30, "longbench_v2": 30}
LONG_BENCH_CATEGORY_ALLOCATION = {
    "single_multi_document_qa": 10,
    "long_icl_dialogue": 10,
    "code_structured": 10,
}
INFINITEBENCH_ALLOCATION: tuple[tuple[str, int, str], ...] = (
    ("longbook_qa_eng", 6, "fake_book_qa"),
    ("longbook_choice_eng", 6, "fake_book_mc"),
    ("longdialogue_qa_eng", 4, "dialogue"),
    ("code_debug", 4, "code_debugging"),
    ("passkey", 4, "pass_key_retrieval"),
    ("number_string", 3, "number_retrieval"),
    ("kv_retrieval", 3, "kv_retrieval"),
)


def manifest_signature(config: Mapping[str, Any], lock: Mapping[str, Any]) -> str:
    """Return the immutable identity of the canonical 40/30/30 source selection."""

    material = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "seed": int(config["project"]["seed"]),
        "dataset_locks": lock.get("datasets", {}),
        "source_locks": lock.get("sources", {}),
        "expected_source_allocation": EXPECTED_SOURCE_ALLOCATION,
        "infinitebench_allocation": INFINITEBENCH_ALLOCATION,
        "longbench_category_allocation": LONG_BENCH_CATEGORY_ALLOCATION,
        "ruler_generator_version": RULER_GENERATOR_VERSION,
        "max_source_context_chars": int(
            config["normalization"].get("max_source_context_chars", 4_000_000)
        ),
    }
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def build_canonical_manifest(
    config: Mapping[str, Any], lock: Mapping[str, Any], *, force: bool = False
) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    canonical_dir = ensure_dir(data_dir / "canonical")
    manifest_path = canonical_dir / "manifest.jsonl"
    metadata_path = canonical_dir / "manifest_metadata.json"
    signature = manifest_signature(config, lock)
    if manifest_path.exists() and metadata_path.exists() and not force:
        reuse_reason = _canonical_manifest_reuse_error(
            manifest_path, metadata_path, signature=signature
        )
        if reuse_reason is None:
            print("[prepare] Reusing canonical manifest with matching immutable signature")
            return manifest_path
        print(f"[prepare] Rebuilding canonical manifest: {reuse_reason}")

    seed = int(config["project"]["seed"])
    records: list[dict[str, Any]] = []

    ruler_records = generate_ruler_records(seed)
    ruler_revision = lock["sources"]["ruler"]["commit_sha"]
    for record in ruler_records:
        record["metadata"]["upstream_commit"] = ruler_revision
    records.extend(ruler_records)

    records.extend(_load_infinitebench(config, lock, seed))
    records.extend(_load_longbench_v2(config, lock, seed))

    counts: dict[str, int] = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    expected = EXPECTED_SOURCE_ALLOCATION
    if counts != expected:
        raise BenchmarkError(f"Canonical allocation mismatch: expected {expected}, got {counts}")
    if len({record["sample_id"] for record in records}) != len(records):
        raise BenchmarkError("Canonical sample IDs are not unique")

    records.sort(key=lambda record: (record["source"], record["sample_id"]))
    write_jsonl(manifest_path, records)
    atomic_write_json(
        metadata_path,
        {
            "format_version": MANIFEST_FORMAT_VERSION,
            "created_at": utc_now(),
            "seed": seed,
            "count": len(records),
            "allocation": counts,
            "manifest_signature": signature,
            "manifest_sha256": sha256_file(manifest_path),
            "lock": lock,
        },
    )
    return manifest_path


def _canonical_manifest_reuse_error(
    manifest_path: Path, metadata_path: Path, *, signature: str
) -> str | None:
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"metadata is unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(metadata, Mapping):
        return "metadata root is not an object"
    if int(metadata.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
        return "manifest format version changed"
    if metadata.get("manifest_signature") != signature:
        return "selection seed, source revisions, allocation, or generator identity changed"
    actual_hash = sha256_file(manifest_path)
    if metadata.get("manifest_sha256") != actual_hash:
        return "manifest checksum no longer matches its metadata"
    try:
        records = list(read_jsonl(manifest_path))
    except (OSError, json.JSONDecodeError, BenchmarkError) as exc:
        return f"manifest is unreadable ({type(exc).__name__}: {exc})"
    if len(records) != 100:
        return f"manifest contains {len(records)} records instead of 100"
    counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    for record in records:
        source = str(record.get("source") or "")
        counts[source] = counts.get(source, 0) + 1
        sample_ids.add(str(record.get("sample_id") or ""))
    if counts != EXPECTED_SOURCE_ALLOCATION:
        return f"source allocation changed: {counts}"
    if len(sample_ids) != len(records) or "" in sample_ids:
        return "sample IDs are missing or duplicated"
    return None


def _load_infinitebench(
    config: Mapping[str, Any], lock: Mapping[str, Any], seed: int
) -> list[dict[str, Any]]:
    try:
        from datasets import Features, Sequence, Value, load_dataset
    except ImportError as exc:
        raise BenchmarkError("The datasets package is required for preparation") from exc

    dataset_lock = lock["datasets"]["infinitebench"]
    repo_id = dataset_lock["repo_id_requested"]
    revision = dataset_lock["commit_sha"]
    cache_dir = config["paths"]["hf_cache_dir"]
    token = os.getenv("HF_TOKEN") or None
    features = Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )

    print(f"[prepare] Loading pinned InfiniteBench dataset {repo_id}@{revision}")
    try:
        dataset_dict = load_dataset(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            features=features,
            streaming=True,
        )
    except Exception:
        # Some future Datasets versions may infer the schema more reliably than
        # an explicit Features declaration. Retain a compatibility fallback.
        dataset_dict = load_dataset(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            streaming=True,
        )

    available = set(dataset_dict.keys())
    records: list[dict[str, Any]] = []
    for split, count, category in INFINITEBENCH_ALLOCATION:
        if split not in available:
            raise BenchmarkError(
                f"InfiniteBench split '{split}' is unavailable; found {sorted(available)}"
            )
        selected = _select_k(
            dataset_dict[split],
            count,
            seed=seed,
            namespace=f"infinitebench:{split}",
            id_getter=lambda row: row.get("id"),
        )
        if len(selected) != count:
            raise BenchmarkError(
                f"InfiniteBench split '{split}' supplied {len(selected)} rows, expected {count}"
            )
        for row in selected:
            original_id = str(row.get("id"))
            options = _as_string_list(row.get("options"))
            question = str(row.get("input") or "").strip()
            instruction = _format_instruction(question, options)
            answer_values = _as_string_list(row.get("answer"))
            records.append(
                {
                    "sample_id": f"infinitebench-{split}-{original_id}",
                    "source": "infinitebench",
                    "task": split,
                    "context": str(row.get("context") or ""),
                    "instruction": instruction,
                    "answer": answer_values,
                    "metadata": {
                        "original_id": original_id,
                        "category": category,
                        "split": split,
                        "dataset_repo": dataset_lock["repo_id_resolved"],
                        "dataset_commit": revision,
                    },
                }
            )
    return records


def _load_longbench_v2(
    config: Mapping[str, Any], lock: Mapping[str, Any], seed: int
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise BenchmarkError("The datasets package is required for preparation") from exc

    dataset_lock = lock["datasets"]["longbench_v2"]
    repo_id = dataset_lock["repo_id_requested"]
    revision = dataset_lock["commit_sha"]
    cache_dir = config["paths"]["hf_cache_dir"]
    token = os.getenv("HF_TOKEN") or None
    print(f"[prepare] Loading pinned LongBench v2 dataset {repo_id}@{revision}")
    dataset = load_dataset(
        repo_id,
        revision=revision,
        split="train",
        cache_dir=cache_dir,
        token=token,
        streaming=True,
    )

    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        "single_multi_document_qa": [],
        "long_icl_dialogue": [],
        "code_structured": [],
    }
    seen_ids: dict[str, set[str]] = {category: set() for category in heaps}
    for row in dataset:
        row_dict = dict(row)
        original_id = str(row_dict.get("_id") or row_dict.get("id") or "").strip()
        if not original_id:
            original_id = sha256_text(
                json.dumps(row_dict, sort_keys=True, default=str, separators=(",", ":"))
            )[:24]
            row_dict["_id"] = original_id
        category = _longbench_category(row_dict)
        if original_id in seen_ids[category]:
            continue
        seen_ids[category].add(original_id)
        _heap_add(
            heaps[category],
            row_dict,
            k=LONG_BENCH_CATEGORY_ALLOCATION[category],
            score=stable_int(seed, "longbench_v2", category, original_id),
            key=original_id,
        )

    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, heap in heaps.items():
        selected = [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]
        required = LONG_BENCH_CATEGORY_ALLOCATION[category]
        if len(selected) != required:
            raise BenchmarkError(
                f"LongBench v2 category '{category}' supplied only {len(selected)} distinct "
                f"classified rows; {required} are required. Refusing to substitute rows from "
                "another category."
            )
        selected_by_category[category] = selected

    records: list[dict[str, Any]] = []
    max_chars = int(config["normalization"].get("max_source_context_chars", 4_000_000))
    for category, selected in selected_by_category.items():
        for row in selected:
            original_id = str(row.get("_id") or row.get("id") or "")
            choices = [
                str(row.get(f"choice_{letter}") or "")
                for letter in ("A", "B", "C", "D")
            ]
            question = str(row.get("question") or "").strip()
            instruction = _format_instruction(question, choices, labels=("A", "B", "C", "D"))
            context = str(row.get("context") or "")
            original_chars = len(context)
            if len(context) > max_chars:
                context = context[:max_chars]
            records.append(
                {
                    "sample_id": f"longbench-v2-{original_id}",
                    "source": "longbench_v2",
                    "task": category,
                    "context": context,
                    "instruction": instruction,
                    "answer": str(row.get("answer") or ""),
                    "metadata": {
                        "original_id": original_id,
                        "category": category,
                        "domain": row.get("domain"),
                        "sub_domain": row.get("sub_domain"),
                        "difficulty": row.get("difficulty"),
                        "length": row.get("length"),
                        "dataset_repo": dataset_lock["repo_id_resolved"],
                        "dataset_commit": revision,
                        "original_context_chars": original_chars,
                        "canonical_context_chars": len(context),
                        "context_char_truncated": original_chars != len(context),
                    },
                }
            )
    return records


def _select_k(
    rows: Iterable[Mapping[str, Any]],
    k: int,
    *,
    seed: int,
    namespace: str,
    id_getter,
) -> list[dict[str, Any]]:
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        row_dict = dict(row)
        key = str(id_getter(row_dict) if id_getter(row_dict) is not None else index)
        score = stable_int(seed, namespace, key)
        _heap_add(heap, row_dict, k=k, score=score, key=key)
    # Heap stores the selected rows with the largest negative values at the
    # front; sorting by -negative score restores ascending deterministic score.
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _heap_add(
    heap: list[tuple[int, str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    k: int,
    score: int,
    key: str,
) -> None:
    entry = (-score, key, row)
    if len(heap) < k:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _longbench_category(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "") for key in ("domain", "sub_domain", "task", "category")
    ).lower()
    if any(keyword in text for keyword in ("code", "structured", "table", "repository")):
        return "code_structured"
    if any(
        keyword in text
        for keyword in ("in-context", "in context", "icl", "dialog", "many-shot", "history")
    ):
        return "long_icl_dialogue"
    return "single_multi_document_qa"


def _format_instruction(
    question: str,
    options: list[str],
    *,
    labels: tuple[str, ...] | None = None,
) -> str:
    parts = ["Task instruction:", question]
    clean_options = [option for option in options if option]
    if clean_options:
        if labels is None:
            labels = tuple(chr(ord("A") + index) for index in range(len(clean_options)))
        parts.append("Answer choices:")
        parts.extend(f"{label}. {option}" for label, option in zip(labels, clean_options))
        parts.append("Return the best answer. Do not add an explanation.")
    else:
        parts.append("Return only the requested answer. Do not add an explanation.")
    return "\n".join(part for part in parts if part)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]
