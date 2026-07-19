from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .normalize import encode, load_pinned_tokenizer, preparation_signature
from .util import (
    BenchmarkError,
    atomic_write_json,
    load_json,
    read_jsonl,
    sha256_file,
    sha256_token_ids,
    utc_now,
)


def validate_prepared_dataset(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    tokenizer=None,
    write_report: bool = True,
) -> dict[str, Any]:
    data_dir = Path(config["paths"]["data_dir"])
    prepared_dir = data_dir / "prepared"
    metadata_path = prepared_dir / "dataset_metadata.json"
    cold_path = prepared_dir / "cold.jsonl"
    warm_path = prepared_dir / "warm_shared.jsonl"
    warmups_path = prepared_dir / "warmup_prefixes.jsonl"
    prefixes_path = prepared_dir / "shared_prefixes.jsonl"
    required = [metadata_path, cold_path, warm_path, warmups_path, prefixes_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise BenchmarkError(
            "Prepared dataset is incomplete. Run './bench prepare'. Missing: " + ", ".join(missing)
        )

    metadata = load_json(metadata_path)
    manifest_path = data_dir / "canonical" / "manifest.jsonl"
    if not manifest_path.exists():
        raise BenchmarkError(
            "Canonical manifest is missing. Run './bench prepare --force-prepare'."
        )
    manifest_metadata = metadata.get("manifest", {})
    expected_manifest_hash = (
        manifest_metadata.get("sha256")
        if isinstance(manifest_metadata, Mapping)
        else None
    )
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise BenchmarkError(
            "Prepared prompts do not reference the current canonical manifest checksum. "
            "Run './bench prepare --force-prepare'."
        )

    expected_signature = preparation_signature(config, lock)
    if metadata.get("preparation_signature") != expected_signature:
        raise BenchmarkError(
            "Prepared prompts do not match the current config/lock. Run './bench prepare --force-prepare'."
        )

    for key, path in (
        ("cold", cold_path),
        ("warm_shared", warm_path),
        ("warmups", warmups_path),
        ("prefixes", prefixes_path),
    ):
        expected_hash = metadata["files"][key]["sha256"]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise BenchmarkError(
                f"Prepared file checksum mismatch for {path}: expected {expected_hash}, got {actual_hash}"
            )

    tokenizer = tokenizer or load_pinned_tokenizer(config, lock)
    target = int(config["project"]["input_tokens"])
    shared_target = int(config["project"]["shared_prefix_tokens"])
    expected_samples = 100
    expected_groups = int(config["project"]["warm_groups"])
    first_tokens = int(config["normalization"].get("validation_first_tokens", 256))

    prefix_by_group: dict[str, dict[str, Any]] = {}
    for prefix_record in read_jsonl(prefixes_path):
        group_id = str(prefix_record["group_id"])
        ids = encode(tokenizer, prefix_record["prefix"])
        if len(ids) != shared_target:
            raise BenchmarkError(
                f"Shared prefix {group_id} has {len(ids)} tokens, expected {shared_target}"
            )
        token_hash = sha256_token_ids(ids)
        if token_hash != prefix_record["prefix_token_sha256"]:
            raise BenchmarkError(f"Shared prefix hash mismatch for {group_id}")
        prefix_by_group[group_id] = {**prefix_record, "ids": ids}
    if len(prefix_by_group) != expected_groups:
        raise BenchmarkError(
            f"Expected {expected_groups} shared prefix groups, found {len(prefix_by_group)}"
        )

    cold_count = 0
    cold_ids: set[str] = set()
    first_prefix_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    for record in read_jsonl(cold_path):
        prompt_ids = encode(tokenizer, record["prompt"])
        if len(prompt_ids) != target:
            raise BenchmarkError(
                f"Cold prompt {record['sample_id']} has {len(prompt_ids)} tokens, expected {target}"
            )
        prefix_hash = sha256_token_ids(prompt_ids[:first_tokens])
        if prefix_hash in first_prefix_hashes:
            raise BenchmarkError(
                f"Cold first-{first_tokens}-token collision at {record['sample_id']}"
            )
        first_prefix_hashes.add(prefix_hash)
        cold_ids.add(str(record["sample_id"]))
        source_counts[str(record["source"])] += 1
        cold_count += 1

    if cold_count != expected_samples:
        raise BenchmarkError(f"Expected {expected_samples} cold prompts, found {cold_count}")
    if len(cold_ids) != cold_count:
        raise BenchmarkError("Cold sample IDs are duplicated")
    expected_sources = Counter({"ruler": 40, "infinitebench": 30, "longbench_v2": 30})
    if source_counts != expected_sources:
        raise BenchmarkError(
            f"Cold source allocation mismatch: expected {dict(expected_sources)}, got {dict(source_counts)}"
        )

    warm_count = 0
    warm_ids: set[str] = set()
    warm_source_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for record in read_jsonl(warm_path):
        sample_id = str(record["sample_id"])
        group_id = str(record["group_id"])
        if group_id not in prefix_by_group:
            raise BenchmarkError(f"Warm prompt {sample_id} references unknown group {group_id}")
        prompt_ids = encode(tokenizer, record["prompt"])
        if len(prompt_ids) != target:
            raise BenchmarkError(
                f"Warm prompt {sample_id} has {len(prompt_ids)} tokens, expected {target}"
            )
        expected_prefix_ids = prefix_by_group[group_id]["ids"]
        if prompt_ids[:shared_target] != expected_prefix_ids:
            raise BenchmarkError(f"Warm prompt {sample_id} does not share the exact group prefix")
        if sha256_token_ids(prompt_ids[:shared_target]) != record["prefix_token_sha256"]:
            raise BenchmarkError(f"Warm prompt {sample_id} prefix hash mismatch")
        warm_ids.add(sample_id)
        warm_source_counts[str(record["source"])] += 1
        group_counts[group_id] += 1
        warm_count += 1

    if warm_count != expected_samples:
        raise BenchmarkError(f"Expected {expected_samples} warm prompts, found {warm_count}")
    if len(warm_ids) != warm_count:
        raise BenchmarkError("Warm sample IDs are duplicated")
    if warm_source_counts != expected_sources:
        raise BenchmarkError(
            "Warm source allocation mismatch: "
            f"expected {dict(expected_sources)}, got {dict(warm_source_counts)}"
        )
    if warm_ids != cold_ids:
        raise BenchmarkError("Cold and warm suites do not contain the same canonical sample IDs")
    expected_per_group = expected_samples // expected_groups
    if set(group_counts.values()) != {expected_per_group}:
        raise BenchmarkError(
            f"Warm group sizes must all be {expected_per_group}; got {dict(group_counts)}"
        )

    warmup_count = 0
    for record in read_jsonl(warmups_path):
        group_id = str(record["group_id"])
        if group_id not in prefix_by_group:
            raise BenchmarkError(f"Warm-up references unknown group {group_id}")
        prompt_ids = encode(tokenizer, record["prompt"])
        if prompt_ids[:shared_target] != prefix_by_group[group_id]["ids"]:
            raise BenchmarkError(f"Warm-up prompt for {group_id} does not preserve the prefix")
        warmup_count += 1
    if warmup_count != expected_groups:
        raise BenchmarkError(f"Expected {expected_groups} warm-up prompts, found {warmup_count}")

    report = {
        "validated_at": utc_now(),
        "valid": True,
        "preparation_signature": expected_signature,
        "cold_prompts": cold_count,
        "warm_prompts": warm_count,
        "warmup_prompts": warmup_count,
        "shared_prefixes": len(prefix_by_group),
        "prompt_tokens": target,
        "shared_prefix_tokens": shared_target,
        "first_prefix_uniqueness_tokens": first_tokens,
        "source_allocation": dict(source_counts),
        "warm_source_allocation": dict(warm_source_counts),
        "canonical_manifest_sha256": actual_manifest_hash,
        "group_counts": dict(sorted(group_counts.items())),
    }
    if write_report:
        atomic_write_json(prepared_dir / "validation_report.json", report)
    print(
        f"[validate] PASS: {cold_count} cold + {warm_count} warm prompts; "
        f"all prompts exactly {target:,} tokens"
    )
    return report
