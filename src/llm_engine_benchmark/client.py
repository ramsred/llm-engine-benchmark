from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp

from .normalize import encode
from .util import (
    BenchmarkError,
    atomic_write_json,
    ensure_dir,
    percentile,
    read_jsonl,
    sha256_text,
    utc_now,
    write_jsonl,
)


@dataclass(frozen=True)
class ClientRunOptions:
    base_url: str
    model: str
    engine: str
    cache_mode: str
    concurrency: int
    output_tokens: int
    request_timeout_seconds: float
    request_extra: Mapping[str, Any]
    save_outputs: bool = True
    require_server_token_usage: bool = True
    api_key: str | None = None


def run_benchmark_client(
    *,
    records_path: str | Path,
    run_dir: str | Path,
    tokenizer,
    options: ClientRunOptions,
    sample_ids: Sequence[str] | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    records = list(read_jsonl(records_path))
    if sample_ids is not None:
        by_id = {str(record.get("sample_id")): record for record in records}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise BenchmarkError(f"Selected sample IDs are missing from {records_path}: {missing}")
        records = [by_id[sample_id] for sample_id in sample_ids]
        if sample_limit is not None and sample_limit != len(records):
            raise BenchmarkError(
                "sample_limit must equal the explicit selected sample ID count"
            )
        sample_limit = None
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise BenchmarkError("No benchmark records were selected")
    return asyncio.run(
        _run_benchmark_async(
            records=records,
            records_path=Path(records_path),
            run_dir=Path(run_dir),
            tokenizer=tokenizer,
            options=options,
        )
    )


async def _run_benchmark_async(
    *,
    records: list[dict[str, Any]],
    records_path: Path,
    run_dir: Path,
    tokenizer,
    options: ClientRunOptions,
) -> dict[str, Any]:
    ensure_dir(run_dir)
    timeout = aiohttp.ClientTimeout(
        total=options.request_timeout_seconds,
        connect=min(60.0, options.request_timeout_seconds),
        sock_read=options.request_timeout_seconds,
    )
    connector = aiohttp.TCPConnector(limit=max(options.concurrency * 2, 8), force_close=False)
    semaphore = asyncio.Semaphore(options.concurrency)
    wall_started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _bounded_request(
                    semaphore=semaphore,
                    session=session,
                    record=record,
                    request_index=index,
                    tokenizer=tokenizer,
                    options=options,
                    wall_origin=wall_started,
                )
            )
            for index, record in enumerate(records)
        ]
        results = await asyncio.gather(*tasks)
    wall_finished = time.perf_counter()

    results.sort(key=lambda result: int(result["request_index"]))
    timings_path = run_dir / "request_timings.jsonl"
    serializable_results = []
    for result in results:
        output_text = result.pop("_output_text", "")
        if options.save_outputs:
            result["output_text"] = output_text
        result["output_sha256"] = sha256_text(output_text)
        serializable_results.append(result)
    write_jsonl(timings_path, serializable_results)

    aggregate = _aggregate_results(
        serializable_results,
        wall_seconds=wall_finished - wall_started,
        options=options,
    )
    aggregate["created_at"] = utc_now()
    aggregate["records_path"] = str(records_path.resolve())
    aggregate["request_timings_path"] = str(timings_path.resolve())
    aggregate["sample_ids"] = [result["sample_id"] for result in serializable_results]
    atomic_write_json(run_dir / "client_results.json", aggregate)
    return aggregate


async def _bounded_request(
    *,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    record: Mapping[str, Any],
    request_index: int,
    tokenizer,
    options: ClientRunOptions,
    wall_origin: float,
) -> dict[str, Any]:
    queued_at = time.perf_counter()
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await _stream_completion(
                session=session,
                base_url=options.base_url,
                model=options.model,
                prompt=str(record["prompt"]),
                max_tokens=options.output_tokens,
                request_extra=options.request_extra,
                api_key=options.api_key,
                tokenizer=tokenizer,
            )
            ended = time.perf_counter()
            output_text = response["text"]
            output_ids = response["token_ids"]
            token_times = response["token_times_seconds"]
            first_token_seconds = response["first_token_seconds"]
            request_elapsed_seconds = float(response["request_elapsed_seconds"])
            retokenized_output_tokens = len(output_ids)
            server_reported_output_tokens = response.get("server_reported_completion_tokens")
            server_reported_prompt_tokens = response.get("server_reported_prompt_tokens")
            server_reported_cached_prompt_tokens = response.get(
                "server_reported_cached_prompt_tokens"
            )
            actual_output_tokens = (
                int(server_reported_output_tokens)
                if server_reported_output_tokens is not None
                else retokenized_output_tokens
            )
            output_token_count_source = (
                "server_usage"
                if server_reported_output_tokens is not None
                else "retokenized_text"
            )
            itls = [
                max(0.0, token_times[index] - token_times[index - 1])
                for index in range(1, len(token_times))
            ]
            tpot = None
            if first_token_seconds is not None and actual_output_tokens > 1:
                tpot = max(
                    0.0,
                    (request_elapsed_seconds - first_token_seconds)
                    / (actual_output_tokens - 1),
                )
            return {
                "request_index": request_index,
                "sample_id": str(record["sample_id"]),
                "source": str(record.get("source") or "unknown"),
                "task": str(record.get("task") or "unknown"),
                "group_id": record.get("group_id"),
                "status": "ok",
                "error": None,
                "queued_seconds": max(0.0, started - queued_at),
                "request_start_offset_seconds": started - wall_origin,
                "request_end_offset_seconds": ended - wall_origin,
                "ttft_seconds": first_token_seconds,
                "e2e_seconds": request_elapsed_seconds,
                "tpot_seconds": tpot,
                "itl_seconds": itls,
                "itl_p50_seconds": percentile(itls, 50),
                "itl_p95_seconds": percentile(itls, 95),
                "input_tokens": int(record.get("prompt_tokens", 0)),
                "expected_output_tokens": options.output_tokens,
                "actual_output_tokens": actual_output_tokens,
                "output_token_count_source": output_token_count_source,
                "server_reported_output_tokens": server_reported_output_tokens,
                "retokenized_output_tokens": retokenized_output_tokens,
                "server_reported_prompt_tokens": server_reported_prompt_tokens,
                "server_reported_cached_prompt_tokens": (
                    server_reported_cached_prompt_tokens
                ),
                "finish_reason": response.get("finish_reason"),
                "stream_events": response["stream_events"],
                "token_event_times_seconds": token_times,
                "prefix_token_sha256": record.get("prefix_token_sha256"),
                "_output_text": output_text,
            }
        except Exception as exc:
            ended = time.perf_counter()
            return {
                "request_index": request_index,
                "sample_id": str(record["sample_id"]),
                "source": str(record.get("source") or "unknown"),
                "task": str(record.get("task") or "unknown"),
                "group_id": record.get("group_id"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "queued_seconds": max(0.0, started - queued_at),
                "request_start_offset_seconds": started - wall_origin,
                "request_end_offset_seconds": ended - wall_origin,
                "ttft_seconds": None,
                "e2e_seconds": ended - started,
                "tpot_seconds": None,
                "itl_seconds": [],
                "itl_p50_seconds": None,
                "itl_p95_seconds": None,
                "input_tokens": int(record.get("prompt_tokens", 0)),
                "expected_output_tokens": options.output_tokens,
                "actual_output_tokens": 0,
                "output_token_count_source": None,
                "server_reported_output_tokens": None,
                "retokenized_output_tokens": 0,
                "server_reported_prompt_tokens": None,
                "server_reported_cached_prompt_tokens": None,
                "finish_reason": None,
                "stream_events": 0,
                "token_event_times_seconds": [],
                "prefix_token_sha256": record.get("prefix_token_sha256"),
                "_output_text": "",
            }


async def _stream_completion(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_extra: Mapping[str, Any],
    api_key: str | None,
    tokenizer,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": True,
        "n": 1,
        "echo": False,
    }
    payload.update(dict(request_extra))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/v1/completions"
    request_started = time.perf_counter()
    async with session.post(url, json=payload, headers=headers) as response:
        if response.status < 200 or response.status >= 300:
            body = (await response.text())[:4000]
            raise BenchmarkError(f"HTTP {response.status} from {url}: {body}")

        text = ""
        token_ids: list[int] = []
        token_times: list[float] = []
        first_token_seconds: float | None = None
        finish_reason: str | None = None
        server_reported_completion_tokens: int | None = None
        server_reported_prompt_tokens: int | None = None
        server_reported_cached_prompt_tokens: int | None = None
        stream_events = 0
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid SSE JSON: {data[:500]}") from exc
            if "error" in event:
                raise BenchmarkError(f"Server stream error: {event['error']}")
            usage = event.get("usage")
            if isinstance(usage, Mapping):
                completion_tokens = _nonnegative_int(usage.get("completion_tokens"))
                if completion_tokens is not None:
                    server_reported_completion_tokens = completion_tokens
                prompt_tokens = _nonnegative_int(
                    usage.get("prompt_tokens", usage.get("input_tokens"))
                )
                if prompt_tokens is not None:
                    server_reported_prompt_tokens = prompt_tokens
                details = usage.get("prompt_tokens_details")
                if not isinstance(details, Mapping):
                    details = usage.get("input_tokens_details")
                cached_tokens = None
                if isinstance(details, Mapping):
                    cached_tokens = _nonnegative_int(details.get("cached_tokens"))
                if cached_tokens is None:
                    cached_tokens = _nonnegative_int(usage.get("cached_tokens"))
                if cached_tokens is not None:
                    server_reported_cached_prompt_tokens = cached_tokens
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta_text = choice.get("text")
            if delta_text is None and isinstance(choice.get("delta"), dict):
                delta_text = choice["delta"].get("content")
            if delta_text:
                stream_events += 1
                now_elapsed = time.perf_counter() - request_started
                text += str(delta_text)
                new_ids = encode(tokenizer, text)
                common = _longest_common_prefix(token_ids, new_ids)
                token_times = token_times[:common] + [now_elapsed] * (len(new_ids) - common)
                token_ids = new_ids
                if first_token_seconds is None and token_ids:
                    first_token_seconds = now_elapsed
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason"))

        # One final tokenization guards against a tokenizer boundary changing in
        # the last streamed chunk.
        final_ids = encode(tokenizer, text)
        common = _longest_common_prefix(token_ids, final_ids)
        final_elapsed = time.perf_counter() - request_started
        token_times = token_times[:common] + [final_elapsed] * (len(final_ids) - common)
        token_ids = final_ids
        if token_ids and first_token_seconds is None:
            first_token_seconds = final_elapsed
        return {
            "text": text,
            "token_ids": token_ids,
            "token_times_seconds": token_times,
            "first_token_seconds": first_token_seconds,
            "finish_reason": finish_reason,
            "stream_events": stream_events,
            "request_elapsed_seconds": final_elapsed,
            "server_reported_completion_tokens": server_reported_completion_tokens,
            "server_reported_prompt_tokens": server_reported_prompt_tokens,
            "server_reported_cached_prompt_tokens": (
                server_reported_cached_prompt_tokens
            ),
        }


def _nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    options: ClientRunOptions,
) -> dict[str, Any]:
    successes = [result for result in results if result["status"] == "ok"]
    failures = [result for result in results if result["status"] != "ok"]
    ttfts = [float(result["ttft_seconds"]) for result in successes if result["ttft_seconds"] is not None]
    e2e = [float(result["e2e_seconds"]) for result in successes]
    tpots = [float(result["tpot_seconds"]) for result in successes if result["tpot_seconds"] is not None]
    itls = [float(value) for result in successes for value in result.get("itl_seconds", [])]
    logical_input_tokens = sum(int(result["input_tokens"]) for result in successes)
    output_tokens = sum(int(result["actual_output_tokens"]) for result in successes)
    expected_output_total = len(results) * options.output_tokens

    prompt_usage_rows = [
        result
        for result in successes
        if result.get("server_reported_prompt_tokens") is not None
    ]
    cache_usage_rows = [
        result
        for result in successes
        if result.get("server_reported_prompt_tokens") is not None
        and result.get("server_reported_cached_prompt_tokens") is not None
    ]
    prompt_usage_total = sum(
        int(result["server_reported_prompt_tokens"]) for result in prompt_usage_rows
    )
    cached_prompt_usage_total = sum(
        int(result["server_reported_cached_prompt_tokens"])
        for result in cache_usage_rows
    )
    cache_prompt_usage_total = sum(
        int(result["server_reported_prompt_tokens"]) for result in cache_usage_rows
    )
    invalid_cache_rows = [
        result
        for result in cache_usage_rows
        if int(result["server_reported_cached_prompt_tokens"])
        > int(result["server_reported_prompt_tokens"])
    ]
    full_cache_usage_coverage = bool(successes) and len(cache_usage_rows) == len(successes)

    if full_cache_usage_coverage and not invalid_cache_rows:
        computed_input_estimate: int | None = sum(
            int(result["server_reported_prompt_tokens"])
            - int(result["server_reported_cached_prompt_tokens"])
            for result in cache_usage_rows
        )
        computed_source = "server_usage_cache_details"
        computed_label = (
            "Observed uncached prompt tokens from server-reported prompt/cache usage details."
        )
    elif options.cache_mode == "cold":
        computed_input_estimate = logical_input_tokens
        computed_source = "cold_protocol_estimate"
        computed_label = (
            "Protocol estimate: all submitted prompt tokens are treated as computed after a fresh "
            "server start and unrelated runtime warm-up."
        )
    elif options.cache_mode == "warm_shared":
        # The orchestrator knows the exact shared-prefix/unique-suffix protocol
        # and fills the lower-bound estimate after this neutral client returns.
        computed_input_estimate = None
        computed_source = None
        computed_label = "Set by the orchestrator from the prepared warm-prefix protocol."
    else:
        computed_input_estimate = None
        computed_source = None
        computed_label = "Cache-capacity dependent; not inferred without complete server usage details."

    validation_errors: list[str] = []
    validation_warnings: list[str] = []
    if failures:
        validation_errors.append(f"{len(failures)} request(s) failed")
    missing_ttft = [result for result in successes if result.get("ttft_seconds") is None]
    missing_server_output_usage = [
        result
        for result in successes
        if result.get("server_reported_output_tokens") is None
    ]
    if options.require_server_token_usage and missing_server_output_usage:
        validation_errors.append(
            f"{len(missing_server_output_usage)} successful request(s) lacked authoritative server completion-token usage"
        )
    if missing_ttft:
        validation_errors.append(f"{len(missing_ttft)} successful request(s) had no streamed token")
    usage_disagreements = [
        result
        for result in successes
        if result.get("server_reported_output_tokens") is not None
        and int(result["server_reported_output_tokens"])
        != int(result.get("retokenized_output_tokens", 0))
    ]
    if usage_disagreements:
        validation_warnings.append(
            f"{len(usage_disagreements)} request(s) had server-usage/output-text token-count "
            "disagreements; server usage is authoritative and decoded-text re-tokenization is "
            "retained as a diagnostic"
        )
    retokenized_output_fallbacks = [
        result
        for result in successes
        if result.get("output_token_count_source") == "retokenized_text"
    ]
    if retokenized_output_fallbacks:
        validation_warnings.append(
            "Streamed server completion-token usage was unavailable for "
            f"{len(retokenized_output_fallbacks)}/{len(successes)} successful request(s); "
            "exact-output validation used decoded-text re-tokenization for those requests."
        )
    prompt_count_disagreements = [
        result
        for result in prompt_usage_rows
        if int(result["server_reported_prompt_tokens"]) != int(result["input_tokens"])
    ]
    if prompt_count_disagreements:
        validation_errors.append(
            f"{len(prompt_count_disagreements)} request(s) had server-reported prompt token counts "
            "different from the prepared canonical count"
        )
    if invalid_cache_rows:
        validation_errors.append(
            f"{len(invalid_cache_rows)} request(s) reported more cached prompt tokens than total "
            "prompt tokens"
        )
    if 0 < len(prompt_usage_rows) < len(successes):
        validation_warnings.append(
            "Server-reported prompt token usage was available for only "
            f"{len(prompt_usage_rows)}/{len(successes)} successful request(s)."
        )
    if 0 < len(cache_usage_rows) < len(successes):
        validation_warnings.append(
            "Server-reported cache token details were available for only "
            f"{len(cache_usage_rows)}/{len(successes)} successful request(s)."
        )
    mismatches = [
        result
        for result in successes
        if int(result["actual_output_tokens"]) != int(result["expected_output_tokens"])
    ]
    if mismatches:
        validation_errors.append(
            f"{len(mismatches)} request(s) did not generate exactly {options.output_tokens} tokens"
        )
    if output_tokens != expected_output_total:
        validation_errors.append(
            f"generated token total {output_tokens} != required {expected_output_total}"
        )
    valid = not validation_errors and len(successes) == len(results)

    return {
        "valid": valid,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "engine": options.engine,
        "cache_mode": options.cache_mode,
        "base_url": options.base_url,
        "model": options.model,
        "concurrency": options.concurrency,
        "require_server_token_usage": options.require_server_token_usage,
        "sample_count": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "measured_wall_time_seconds": wall_seconds,
        "request_throughput_per_second": len(successes) / wall_seconds if wall_seconds > 0 else None,
        "logical_input_tokens": logical_input_tokens,
        "logical_input_throughput_tokens_per_second": (
            logical_input_tokens / wall_seconds if wall_seconds > 0 else None
        ),
        "computed_input_tokens_estimate": computed_input_estimate,
        "computed_input_token_count_source": computed_source,
        "computed_input_throughput_tokens_per_second": (
            computed_input_estimate / wall_seconds
            if computed_input_estimate is not None and wall_seconds > 0
            else None
        ),
        "computed_input_estimate_note": computed_label,
        "server_reported_prompt_tokens_total": prompt_usage_total,
        "server_reported_prompt_token_coverage_requests": len(prompt_usage_rows),
        "server_reported_cached_prompt_tokens_total": cached_prompt_usage_total,
        "cache_report_prompt_tokens_total": cache_prompt_usage_total,
        "cache_report_coverage_requests": len(cache_usage_rows),
        "cache_report_coverage_fraction": (
            len(cache_usage_rows) / len(successes) if successes else None
        ),
        "cache_hit_ratio": (
            cached_prompt_usage_total / cache_prompt_usage_total
            if cache_prompt_usage_total > 0
            else None
        ),
        "generated_output_tokens": output_tokens,
        "expected_generated_output_tokens": expected_output_total,
        "output_throughput_tokens_per_second": output_tokens / wall_seconds if wall_seconds > 0 else None,
        "ttft_seconds": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "mean": sum(ttfts) / len(ttfts) if ttfts else None,
        },
        "e2e_seconds": {
            "p50": percentile(e2e, 50),
            "p95": percentile(e2e, 95),
            "mean": sum(e2e) / len(e2e) if e2e else None,
        },
        "tpot_seconds": {
            "p50": percentile(tpots, 50),
            "p95": percentile(tpots, 95),
            "mean": sum(tpots) / len(tpots) if tpots else None,
        },
        "itl_seconds": {
            "p50": percentile(itls, 50),
            "p95": percentile(itls, 95),
            "mean": sum(itls) / len(itls) if itls else None,
            "observations": len(itls),
            "note": (
                "Token timestamps are derived from streamed text events. Multiple tokens in one "
                "SSE event share an observation timestamp rather than an invented interpolation."
            ),
        },
        "failed_sample_ids": [result["sample_id"] for result in failures],
        "missing_ttft_sample_ids": [result["sample_id"] for result in missing_ttft],
        "output_mismatch_sample_ids": [result["sample_id"] for result in mismatches],
        "usage_disagreement_sample_ids": [
            result["sample_id"] for result in usage_disagreements
        ],
        "missing_server_output_usage_sample_ids": [
            result["sample_id"] for result in missing_server_output_usage
        ],
        "prompt_token_mismatch_sample_ids": [
            result["sample_id"] for result in prompt_count_disagreements
        ],
        "cache_detail_invalid_sample_ids": [
            result["sample_id"] for result in invalid_cache_rows
        ],
        "output_count_sources": {
            "server_usage": sum(
                1
                for result in successes
                if result.get("output_token_count_source") == "server_usage"
            ),
            "retokenized_text": sum(
                1
                for result in successes
                if result.get("output_token_count_source") == "retokenized_text"
            ),
        },
    }


def send_warmup_requests(
    *,
    base_url: str,
    model: str,
    prompts: Sequence[tuple[str, str]],
    tokenizer,
    request_extra: Mapping[str, Any],
    timeout_seconds: float,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _send_warmups_async(
            base_url=base_url,
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            request_extra=request_extra,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
    )


async def _send_warmups_async(
    *,
    base_url: str,
    model: str,
    prompts: Sequence[tuple[str, str]],
    tokenizer,
    request_extra: Mapping[str, Any],
    timeout_seconds: float,
    api_key: str | None,
) -> list[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds, sock_read=timeout_seconds)
    results: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for label, prompt in prompts:
            started = time.perf_counter()
            response = await _stream_completion(
                session=session,
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=1,
                request_extra=request_extra,
                api_key=api_key,
                tokenizer=tokenizer,
            )
            results.append(
                {
                    "label": label,
                    "duration_seconds": time.perf_counter() - started,
                    "output_tokens": (
                        response.get("server_reported_completion_tokens")
                        if response.get("server_reported_completion_tokens") is not None
                        else len(response["token_ids"])
                    ),
                    "server_reported_output_tokens": response.get(
                        "server_reported_completion_tokens"
                    ),
                    "retokenized_output_tokens": len(response["token_ids"]),
                    "finish_reason": response.get("finish_reason"),
                    "output_sha256": sha256_text(response["text"]),
                }
            )
    return results
