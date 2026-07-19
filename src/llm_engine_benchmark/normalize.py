from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import relevant_preparation_config
from .util import (
    BenchmarkError,
    atomic_write_json,
    ensure_dir,
    read_jsonl,
    sha256_file,
    sha256_text,
    sha256_token_ids,
    stable_int,
    utc_now,
    write_jsonl,
)


DISTRACTOR_WORDS = (
    "amber atlas axis beacon birch bloom bridge brook canyon cedar circuit cloud copper "
    "delta document drift dune echo element ember engine falcon fern field forest frame "
    "garden glacier granite graph grove harbor heather history hollow horizon index inlet "
    "island ivory jade jasmine junction juniper kernel kestrel key keystone knoll lagoon "
    "lake lantern lichen logic maple matrix meadow mineral moss nectar night node north "
    "novel object ocean olive opal orbit path pattern pebble pine prairie query quartz "
    "queue quill rain record reed ridge river saffron schema signal stone summit thicket "
    "thread tide timber tower union unit update upland value valley vector velvet vista "
    "wander wheat willow wind window xylem xenon yarrow yearling yellow yield zephyr zenith "
    "acorn apricot archive balance channel datum evidence feature gateway interval ledger "
    "marker network packet protocol sample segment system token trace version workload"
).split()


class TokenizerProtocol:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


def load_pinned_tokenizer(config: Mapping[str, Any], lock: Mapping[str, Any]):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BenchmarkError("transformers is required for prompt preparation") from exc

    model = lock["model"]["repo_id"]
    revision = lock["model"]["tokenizer_commit_sha"]
    token = os.getenv("HF_TOKEN") or None
    print(f"[prepare] Loading pinned tokenizer {model}@{revision}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            cache_dir=config["paths"]["hf_cache_dir"],
            token=token,
            use_fast=True,
            trust_remote_code=bool(config["project"].get("trust_remote_code", False)),
        )
    except Exception as exc:
        raise BenchmarkError(f"Unable to load tokenizer {model}@{revision}: {exc}") from exc
    return tokenizer


def preparation_signature(config: Mapping[str, Any], lock: Mapping[str, Any]) -> str:
    material = {
        "config": relevant_preparation_config(config),
        "lock": lock,
        "format_version": 1,
    }
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def prepare_prompt_files(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    manifest_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    data_dir = Path(config["paths"]["data_dir"])
    prepared_dir = ensure_dir(data_dir / "prepared")
    files = {
        "cold": prepared_dir / "cold.jsonl",
        "warm_shared": prepared_dir / "warm_shared.jsonl",
        "warmups": prepared_dir / "warmup_prefixes.jsonl",
        "prefixes": prepared_dir / "shared_prefixes.jsonl",
        "metadata": prepared_dir / "dataset_metadata.json",
    }
    signature = preparation_signature(config, lock)
    current_manifest_hash = sha256_file(manifest_path)
    if not force and all(path.exists() for path in files.values()):
        try:
            existing = json.loads(files["metadata"].read_text(encoding="utf-8"))
            manifest_metadata = existing.get("manifest", {})
            if (
                existing.get("preparation_signature") == signature
                and isinstance(manifest_metadata, Mapping)
                and manifest_metadata.get("sha256") == current_manifest_hash
            ):
                print("[prepare] Reusing prepared prompts with matching immutable signature")
                return files
            print("[prepare] Rebuilding prepared prompts because config, lock, or manifest changed")
        except (json.JSONDecodeError, OSError):
            print("[prepare] Rebuilding prepared prompts because metadata is unreadable")

    records = list(read_jsonl(manifest_path))
    if len(records) != 100:
        raise BenchmarkError(f"Canonical manifest must contain 100 records, found {len(records)}")

    tokenizer = load_pinned_tokenizer(config, lock)
    cold = _build_cold_records(config, tokenizer, records)
    warm, warmups, prefixes = _build_warm_records(config, tokenizer, records)

    write_jsonl(files["cold"], cold)
    write_jsonl(files["warm_shared"], warm)
    write_jsonl(files["warmups"], warmups)
    write_jsonl(files["prefixes"], prefixes)

    metadata = {
        "format_version": 1,
        "created_at": utc_now(),
        "preparation_signature": signature,
        "model": lock["model"],
        "prompt_tokens": int(config["project"]["input_tokens"]),
        "output_tokens": int(config["project"]["output_tokens"]),
        "shared_prefix_tokens": int(config["project"]["shared_prefix_tokens"]),
        "unique_suffix_tokens": int(config["project"]["input_tokens"])
        - int(config["project"]["shared_prefix_tokens"]),
        "warm_groups": int(config["project"]["warm_groups"]),
        "sample_count": len(records),
        "files": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in files.items()
            if key != "metadata"
        },
        "manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": current_manifest_hash,
        },
    }
    atomic_write_json(files["metadata"], metadata)
    return files


def _build_cold_records(
    config: Mapping[str, Any], tokenizer: TokenizerProtocol, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    target = int(config["project"]["input_tokens"])
    output_tokens = int(config["project"]["output_tokens"])
    seed = int(config["project"]["seed"])
    extra = int(config["normalization"].get("extra_source_tokens", 4096))
    prepared: list[dict[str, Any]] = []

    print(f"[prepare] Building {len(records)} cold unique-prefix prompts at {target:,} tokens")
    for index, record in enumerate(records, start=1):
        unique_prelude = _unique_prelude(record["sample_id"], seed)
        original_context = str(record.get("context") or "")
        base_source = unique_prelude + "\n\n" + original_context
        source, distractor_meta = ensure_token_capacity(
            tokenizer,
            base_source,
            required_tokens=target + extra,
            seed=stable_int(seed, "cold", record["sample_id"]),
            label=f"cold-{record['sample_id']}",
            min_chars=int(
                config["normalization"].get("controlled_distractor_min_chars", 640_000)
            ),
        )
        suffix, instruction_tokens = instruction_suffix_with_budget(
            config, tokenizer, record
        )
        prompt, prompt_ids, fit_meta = fit_variable_segment(
            tokenizer,
            prefix="",
            segment_source=source,
            suffix=suffix,
            target_tokens=target,
            label=f"cold:{record['sample_id']}",
        )
        prepared.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "task": record["task"],
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
                "output_tokens": output_tokens,
                "expected_answer": record.get("answer"),
                "metadata": {
                    **record.get("metadata", {}),
                    "cache_mode": "cold",
                    "normalization": {
                        **distractor_meta,
                        **fit_meta,
                        "original_context_chars": len(original_context),
                        "instruction_tokens": instruction_tokens,
                        "instruction_reserve_tokens": int(
                            config["normalization"].get("instruction_reserve_tokens", 1024)
                        ),
                    },
                    "first_256_token_sha256": sha256_token_ids(prompt_ids[:256]),
                    "prompt_sha256": sha256_text(prompt),
                },
            }
        )
        print(f"[prepare] cold {index:03d}/{len(records)} {record['sample_id']}")
    return prepared


def _build_warm_records(
    config: Mapping[str, Any], tokenizer: TokenizerProtocol, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = int(config["project"]["input_tokens"])
    shared_tokens = int(config["project"]["shared_prefix_tokens"])
    output_tokens = int(config["project"]["output_tokens"])
    group_count = int(config["project"]["warm_groups"])
    seed = int(config["project"]["seed"])
    extra = int(config["normalization"].get("extra_source_tokens", 4096))
    if len(records) % group_count != 0:
        raise BenchmarkError(
            f"The canonical sample count ({len(records)}) must be divisible by warm_groups ({group_count})"
        )

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[index % group_count].append(record)

    prepared_by_group: dict[int, list[dict[str, Any]]] = {}
    warmups: list[dict[str, Any]] = []
    prefix_records: list[dict[str, Any]] = []
    print(
        f"[prepare] Building {group_count} shared prefixes at {shared_tokens:,} tokens "
        f"and {len(records)} measured prompts at {target:,} tokens"
    )

    for group_id in range(group_count):
        group_label = f"group-{group_id:02d}"
        shared_source_seed = stable_int(seed, "warm-prefix", group_id)
        shared_base = (
            f"<<<SHARED PREFIX {group_label}>>>\n"
            "The following neutral reference archive is shared byte-for-byte by every request "
            "in this cache group.\n\n"
            + controlled_distractor(
                shared_source_seed,
                int(config["normalization"].get("controlled_distractor_min_chars", 640_000)),
                label=f"shared-{group_label}",
            )
        )
        shared_source, _ = ensure_token_capacity(
            tokenizer,
            shared_base,
            required_tokens=shared_tokens + extra,
            seed=shared_source_seed,
            label=f"shared-{group_label}",
            min_chars=int(
                config["normalization"].get("controlled_distractor_min_chars", 640_000)
            ),
        )
        marker = f"\n\n<<<END OF SHARED PREFIX {group_label}>>>\n"
        prefix_text, prefix_ids, prefix_fit = fit_variable_segment(
            tokenizer,
            prefix="",
            segment_source=shared_source,
            suffix=marker,
            target_tokens=shared_tokens,
            label=f"shared-prefix:{group_label}",
        )
        if len(prefix_ids) != shared_tokens:
            raise BenchmarkError(f"Internal prefix length failure for {group_label}")
        prefix_hash = sha256_token_ids(prefix_ids)
        guard = find_prefix_guard(
            tokenizer,
            prefix_text,
            prefix_ids,
            config["normalization"].get("prefix_boundary_candidates", []),
        )
        prefix_records.append(
            {
                "group_id": group_label,
                "prefix": prefix_text,
                "prefix_tokens": shared_tokens,
                "prefix_token_sha256": prefix_hash,
                "guard": guard,
                "metadata": {"normalization": prefix_fit},
            }
        )

        warmup_prompt = prefix_text + guard + "Warm-up suffix. Reply with READY."
        warmup_ids = encode(tokenizer, warmup_prompt)
        if warmup_ids[:shared_tokens] != prefix_ids:
            raise BenchmarkError(f"Warm-up prompt does not preserve shared prefix for {group_label}")
        warmups.append(
            {
                "group_id": group_label,
                "prompt": warmup_prompt,
                "prompt_tokens": len(warmup_ids),
                "shared_prefix_tokens": shared_tokens,
                "prefix_token_sha256": prefix_hash,
            }
        )

        group_prepared: list[dict[str, Any]] = []
        for member_index, record in enumerate(groups[group_id]):
            unique_header = (
                f"UNIQUE REQUEST {record['sample_id']} / {group_label} / member-{member_index:02d}\n"
            )
            base_unique = guard + unique_header + str(record.get("context") or "")
            unique_source, distractor_meta = ensure_token_capacity(
                tokenizer,
                base_unique,
                required_tokens=(target - shared_tokens) + extra,
                seed=stable_int(seed, "warm-suffix", record["sample_id"]),
                label=f"warm-{record['sample_id']}",
                min_chars=max(
                    120_000,
                    int(
                        config["normalization"].get(
                            "controlled_distractor_min_chars", 640_000
                        )
                    )
                    // 4,
                ),
            )
            suffix, instruction_tokens = instruction_suffix_with_budget(
                config, tokenizer, record
            )
            prompt, full_ids, fit_meta = fit_variable_segment(
                tokenizer,
                prefix=prefix_text,
                segment_source=unique_source,
                suffix=suffix,
                target_tokens=target,
                label=f"warm:{record['sample_id']}",
                preserve_prefix_ids=prefix_ids,
            )
            if full_ids[:shared_tokens] != prefix_ids:
                raise BenchmarkError(
                    f"Measured prompt does not preserve shared prefix for {record['sample_id']}"
                )
            group_prepared.append(
                {
                    "sample_id": record["sample_id"],
                    "source": record["source"],
                    "task": record["task"],
                    "prompt": prompt,
                    "prompt_tokens": len(full_ids),
                    "output_tokens": output_tokens,
                    "expected_answer": record.get("answer"),
                    "group_id": group_label,
                    "shared_prefix_tokens": shared_tokens,
                    "unique_suffix_tokens": target - shared_tokens,
                    "prefix_token_sha256": prefix_hash,
                    "metadata": {
                        **record.get("metadata", {}),
                        "cache_mode": "warm_shared",
                        "group_member_index": member_index,
                        "normalization": {
                            **distractor_meta,
                            **fit_meta,
                            "instruction_tokens": instruction_tokens,
                            "instruction_reserve_tokens": int(
                                config["normalization"].get(
                                    "instruction_reserve_tokens", 1024
                                )
                            ),
                        },
                        "prompt_sha256": sha256_text(prompt),
                    },
                }
            )
            print(
                f"[prepare] warm {group_label} member {member_index + 1:02d}/"
                f"{len(groups[group_id]):02d} {record['sample_id']}"
            )
        prepared_by_group[group_id] = group_prepared

    # Round-robin order prevents one cache group from monopolizing the front or
    # back of the measured request stream.
    measured: list[dict[str, Any]] = []
    members_per_group = len(next(iter(prepared_by_group.values())))
    for member_index in range(members_per_group):
        for group_id in range(group_count):
            measured.append(prepared_by_group[group_id][member_index])
    return measured, warmups, prefix_records


def ensure_token_capacity(
    tokenizer: TokenizerProtocol,
    base_text: str,
    *,
    required_tokens: int,
    seed: int,
    label: str,
    min_chars: int,
) -> tuple[str, dict[str, Any]]:
    base_ids = encode(tokenizer, base_text)
    original_tokens = len(base_ids)
    if original_tokens >= required_tokens:
        return base_text, {
            "original_source_tokens": original_tokens,
            "controlled_distractor_added": False,
            "added_distractor_chars": 0,
        }

    estimated_chars = max(min_chars, (required_tokens - original_tokens) * 7)
    added_parts: list[str] = []
    total_chars = 0
    round_index = 0
    combined = base_text
    while True:
        chunk_chars = estimated_chars if round_index == 0 else max(100_000, estimated_chars // 2)
        chunk = controlled_distractor(
            seed + round_index * 97,
            chunk_chars,
            label=f"{label}-distractor-{round_index}",
        )
        added_parts.append(chunk)
        total_chars += len(chunk)
        combined = base_text + "\n\n" + "\n\n".join(added_parts)
        combined_tokens = len(encode(tokenizer, combined))
        if combined_tokens >= required_tokens:
            return combined, {
                "original_source_tokens": original_tokens,
                "extended_source_tokens": combined_tokens,
                "controlled_distractor_added": True,
                "added_distractor_chars": total_chars,
                "distractor_rounds": round_index + 1,
            }
        round_index += 1
        if round_index > 8:
            raise BenchmarkError(
                f"Unable to extend source '{label}' to {required_tokens} tokens; "
                f"reached {combined_tokens}"
            )


def controlled_distractor(seed: int, minimum_chars: int, *, label: str) -> str:
    lines: list[str] = []
    index = 0
    current_chars = 0
    rng = random.Random(seed)
    word_count = len(DISTRACTOR_WORDS)
    while current_chars < minimum_chars:
        values = [DISTRACTOR_WORDS[rng.randrange(word_count)] for _ in range(22)]
        serial = rng.randrange(1_000_000_000)
        line = (
            f"[{label}:{index:06d}:{serial:09d}] "
            + " ".join(values)
            + ". This controlled distractor sentence preserves lexical variety and provenance."
        )
        lines.append(line)
        current_chars += len(line) + 1
        index += 1
    return "\n".join(lines)


def fit_variable_segment(
    tokenizer: TokenizerProtocol,
    *,
    prefix: str,
    segment_source: str,
    suffix: str,
    target_tokens: int,
    label: str,
    preserve_prefix_ids: Sequence[int] | None = None,
) -> tuple[str, list[int], dict[str, Any]]:
    """Fit a variable source segment while preserving fixed prefix/suffix text.

    The routine operates on token-prefixes of the variable segment, decodes the
    prefix to text, and re-tokenizes the complete saved prompt. A fixed-point
    correction handles tokenizer merges at the segment boundaries. The written
    string is always re-tokenized and checked before being returned.
    """

    segment_ids = encode(tokenizer, segment_source)
    fixed_count = len(encode(tokenizer, prefix + suffix))
    if fixed_count >= target_tokens:
        raise BenchmarkError(
            f"Fixed prefix/suffix for {label} already uses {fixed_count} tokens, "
            f"target is {target_tokens}"
        )
    if len(segment_ids) + fixed_count < target_tokens:
        raise BenchmarkError(
            f"Variable source for {label} is too short: {len(segment_ids)} variable tokens, "
            f"requires approximately {target_tokens - fixed_count}"
        )

    n = max(1, min(len(segment_ids), target_tokens - fixed_count))
    seen: set[int] = set()
    evaluated: dict[int, tuple[str, list[int]]] = {}
    best_below: tuple[int, int, str, list[int]] | None = None
    best_absolute: tuple[int, int, str, list[int]] | None = None

    def evaluate(token_prefix_length: int) -> tuple[str, list[int]]:
        token_prefix_length = max(0, min(len(segment_ids), token_prefix_length))
        if token_prefix_length in evaluated:
            return evaluated[token_prefix_length]
        variable_text = decode(tokenizer, segment_ids[:token_prefix_length])
        full_text = prefix + variable_text + suffix
        full_ids = encode(tokenizer, full_text)
        if preserve_prefix_ids is not None:
            expected = list(preserve_prefix_ids)
            if full_ids[: len(expected)] != expected:
                raise BenchmarkError(
                    f"Tokenizer boundary changed the shared prefix while fitting {label}. "
                    "Try changing normalization.prefix_boundary_candidates."
                )
        evaluated[token_prefix_length] = (full_text, full_ids)
        return full_text, full_ids

    for _ in range(32):
        full_text, full_ids = evaluate(n)
        count = len(full_ids)
        delta = target_tokens - count
        absolute = abs(delta)
        candidate_absolute = (absolute, n, full_text, full_ids)
        if best_absolute is None or candidate_absolute[:2] < best_absolute[:2]:
            best_absolute = candidate_absolute
        if count <= target_tokens:
            candidate_below = (target_tokens - count, n, full_text, full_ids)
            if best_below is None or candidate_below[:2] < best_below[:2]:
                best_below = candidate_below
        if delta == 0:
            return full_text, full_ids, {
                "fit_method": "token_prefix_fixed_point",
                "segment_source_tokens": len(segment_ids),
                "segment_tokens_consumed": n,
                "fit_evaluations": len(evaluated),
            }
        seen.add(n)
        corrected = max(0, min(len(segment_ids), n + delta))
        if corrected == n or corrected in seen:
            break
        n = corrected

    assert best_absolute is not None
    center = best_absolute[1]
    offsets: list[int] = [0]
    for distance in range(1, 65):
        offsets.extend((-distance, distance))
    for offset in offsets:
        candidate_n = center + offset
        if candidate_n < 0 or candidate_n > len(segment_ids) or candidate_n in evaluated:
            continue
        full_text, full_ids = evaluate(candidate_n)
        count = len(full_ids)
        if count == target_tokens:
            return full_text, full_ids, {
                "fit_method": "token_prefix_local_search",
                "segment_source_tokens": len(segment_ids),
                "segment_tokens_consumed": candidate_n,
                "fit_evaluations": len(evaluated),
            }
        if count < target_tokens:
            candidate_below = (target_tokens - count, candidate_n, full_text, full_ids)
            if best_below is None or candidate_below[:2] < best_below[:2]:
                best_below = candidate_below

    if best_below is not None:
        gap, candidate_n, _, _ = best_below
        variable_text = decode(tokenizer, segment_ids[:candidate_n])
        shim_result = _fit_small_shim(
            tokenizer,
            prefix=prefix + variable_text,
            suffix=suffix,
            target_tokens=target_tokens,
            gap_hint=gap,
            label=label,
            preserve_prefix_ids=preserve_prefix_ids,
        )
        if shim_result is not None:
            full_text, full_ids, shim = shim_result
            return full_text, full_ids, {
                "fit_method": "token_prefix_plus_calibration_shim",
                "segment_source_tokens": len(segment_ids),
                "segment_tokens_consumed": candidate_n,
                "fit_evaluations": len(evaluated),
                "calibration_shim": shim,
            }

    closest_count = len(best_absolute[3])
    raise BenchmarkError(
        f"Could not fit {label} to exactly {target_tokens} tokens. Closest count was "
        f"{closest_count}. This tokenizer may require an additional boundary candidate."
    )


def _fit_small_shim(
    tokenizer: TokenizerProtocol,
    *,
    prefix: str,
    suffix: str,
    target_tokens: int,
    gap_hint: int,
    label: str,
    preserve_prefix_ids: Sequence[int] | None,
) -> tuple[str, list[int], str] | None:
    fragments = [
        " §",
        " ¶",
        " Ω",
        " q",
        " z",
        " 7",
        " _",
        "-",
        ".",
        ",",
        ":",
        ";",
        "\n",
        "\u241e",
        " calibration",
        " marker",
    ]

    def check(shim: str) -> tuple[str, list[int], str] | None:
        text = prefix + shim + suffix
        ids = encode(tokenizer, text)
        if preserve_prefix_ids is not None:
            expected = list(preserve_prefix_ids)
            if ids[: len(expected)] != expected:
                return None
        if len(ids) == target_tokens:
            return text, ids, shim
        return None

    for fragment in fragments:
        result = check(fragment)
        if result:
            return result

    # Deterministic varied sequences cover small skipped token-count states
    # without padding with one repeated token.
    max_length = min(max(gap_hint + 12, 12), 96)
    for length in range(2, max_length + 1):
        shim = "".join(fragments[index % len(fragments)] for index in range(length))
        result = check(shim)
        if result:
            return result
    return None


def find_prefix_guard(
    tokenizer: TokenizerProtocol,
    prefix_text: str,
    prefix_ids: Sequence[int],
    configured_candidates: Sequence[str],
) -> str:
    candidates = list(configured_candidates) + [
        "\u241e",
        "\u241f",
        "\n#\n",
        "\n[UNIQUE]\n",
        " | ",
        " ~ ",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        combined = encode(tokenizer, prefix_text + candidate + "UNIQUE REQUEST BOUNDARY")
        if combined[: len(prefix_ids)] == list(prefix_ids) and len(combined) > len(prefix_ids):
            return candidate
    punctuation = "!#$%&*+,-./:;=?@^_|~"
    for left in punctuation:
        for right in punctuation:
            candidate = f"\n{left}{right}\n"
            combined = encode(tokenizer, prefix_text + candidate + "UNIQUE REQUEST BOUNDARY")
            if combined[: len(prefix_ids)] == list(prefix_ids) and len(combined) > len(prefix_ids):
                return candidate
    raise BenchmarkError(
        "Could not find a tokenizer-stable boundary after a 100K shared prefix. "
        "Add candidates under normalization.prefix_boundary_candidates."
    )


def encode(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in token_ids]


def decode(tokenizer: TokenizerProtocol, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids))


def _instruction_suffix(record: Mapping[str, Any]) -> str:
    instruction = str(record.get("instruction") or "").strip()
    return (
        "\n\n===== BENCHMARK TASK (DO NOT IGNORE) =====\n"
        + instruction
        + "\n===== END BENCHMARK TASK =====\n"
    )


def instruction_suffix_with_budget(
    config: Mapping[str, Any],
    tokenizer: TokenizerProtocol,
    record: Mapping[str, Any],
) -> tuple[str, int]:
    suffix = _instruction_suffix(record)
    token_count = len(encode(tokenizer, suffix))
    reserve = int(config["normalization"].get("instruction_reserve_tokens", 1024))
    if reserve <= 0:
        raise BenchmarkError("normalization.instruction_reserve_tokens must be positive")
    if token_count > reserve:
        raise BenchmarkError(
            f"Instruction for {record.get('sample_id', '<unknown>')} uses {token_count} tokens, "
            f"exceeding the configured {reserve}-token reserve"
        )
    return suffix, token_count


def _unique_prelude(sample_id: str, seed: int) -> str:
    words = []
    for index in range(320):
        word = DISTRACTOR_WORDS[stable_int(seed, sample_id, index) % len(DISTRACTOR_WORDS)]
        serial = stable_int(seed, "prelude", sample_id, index) % 100_000
        words.append(f"{word}-{serial:05d}")
    return (
        f"<<<COLD UNIQUE SAMPLE {sample_id}>>>\n"
        + " ".join(words)
        + "\n<<<END UNIQUE PRELUDE>>>"
    )
