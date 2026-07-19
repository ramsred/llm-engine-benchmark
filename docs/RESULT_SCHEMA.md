# Result schema overview

## `client_results.json`

Contains the run acceptance decision, validation errors and warnings, request
counts, measured wall time, logical input/output totals and throughput,
TTFT/E2E/TPOT/ITL summaries, failed sample IDs, and fixed-output validation evidence.

With the default strict policy, missing server completion-token usage rejects the
run; decoded-text re-tokenization remains diagnostic evidence only. The result
also lists `missing_server_output_usage_sample_ids`.

Prompt/cache usage fields include:

- `server_reported_prompt_tokens_total` and its request coverage;
- `server_reported_cached_prompt_tokens_total`;
- `cache_report_coverage_requests` and `cache_report_coverage_fraction`;
- `cache_hit_ratio` when cache details are reported;
- `computed_input_tokens_estimate`, throughput, source, and explanatory note;
- `protocol_unique_suffix_tokens_estimate` for warm shared-prefix runs.

Complete server-reported prompt/cache usage is preferred for observed uncached
prompt work. Cold and warm protocol estimates are clearly labeled fallbacks.

## `request_timings.jsonl`

One row per measured request:

- canonical sample/source/task/group identifiers;
- queue, start, and end timing;
- TTFT, E2E, TPOT, observed token-event timestamps, and ITLs;
- prepared input count and actual/expected output counts;
- server-reported prompt, completion, and cached-prompt counts when exposed;
- the authoritative output-count source and decoded-text re-tokenization count;
- HTTP/stream status, finish reason, and errors;
- output text or output hash, according to configuration;
- warm-prefix SHA when applicable.

## `run_metadata.json`

Contains experiment identity, repetition/order, model and tokenizer commits,
container image/digest, rendered server command, runtime library versions,
matched fairness settings, selected sample IDs, cache protocol, validation
warnings, and accepted/rejected state.


`benchmark_scope` explicitly identifies the experiment as serving-performance
only, and `runtime_warmup_limitation` records that first-request long-context
compilation remains part of measured cold behavior.

## Metrics and telemetry

`metrics_before.prom` and `metrics_after.prom` are raw engine outputs.
`metrics_diff.json` computes numeric deltas and highlights cache/prefix/prefill
metric names without assuming stable metric names across engine versions.
`telemetry.csv`, `host_telemetry.jsonl`, and `cpu_memory.txt` preserve GPU, CPU,
memory, power, clock, and temperature evidence where host tools expose it.

## Report files

`runs.csv` contains every accepted or rejected run. `summary.csv` and the
mode-specific CSVs aggregate only accepted repetitions. Cache report coverage
and cache-hit ratio remain blank when an engine release does not expose usable
per-request cache details; the raw logs and Prometheus deltas remain available.

`report_status.json` compares accepted artifacts with every entry in
`active_experiment.json`. The report is `COMPLETE` only when all planned runs are
present and accepted; otherwise missing and rejected run identifiers are listed
and the Markdown report is prominently marked `INCOMPLETE`.


## Experiment scoping

`results/active_experiment.json` records a deterministic scope signature over the prepared-data identity, matrix, sample count, telemetry/cooldown settings, engine configuration, and resolved Docker image IDs. Report generation includes only run metadata carrying that signature, preventing stale evidence from a previous matrix from being merged silently.
