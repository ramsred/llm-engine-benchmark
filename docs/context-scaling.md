# Phase 1: context-length characterization

This opt-in workflow measures fixed-length cold-prefix requests at bounded client
concurrency. It is **not** mixed-workload production capacity, an accuracy benchmark,
or evidence of P99 reliability. Existing `bench run`, `bench capacity`, preparation,
and reporting code paths are unchanged.

## Experimental contract

| Dimension | Default |
| --- | --- |
| Engine and model | Direct TensorRT-LLM, existing pinned model/tokenizer lock |
| Input lengths | 8,000 / 32,000 / 64,000 / 120,000 tokens, exactly |
| Client concurrency | 1 and 4 |
| Output | 512 tokens; server usage required |
| Server context ceiling | Fixed at 131,072; not resized per input length |
| Prefill budget | Fixed at 2,048; not claimed optimal at every length |
| KV dtype / memory fraction | FP8 / 0.80 |
| Discovery | 20 measured requests, one repetition per cell |
| Warm-up | Two waves of C unique requests, target input length and full output length |
| Run order | Seeded shuffle of eight cells; reversed on even repetitions |

Client-observed overlap is verified during warm-up and measurement. It does not
prove simultaneous GPU execution or that every compilation path is initialized.
Raw warm-up timings are retained for inspection. Warm-up requests are never
counted in the measured report. Nsight captures can include startup/warm-up;
filter the timeline before attributing activity to the measurement interval.

## Preservation and isolation

- Start from an idle Spark1. Do not run other GPU workloads during the experiment.
- The existing lock and canonical manifest are mandatory, checked, and read-only.
  The runner never auto-prepares or refreshes the canonical dataset or lock.
- New prompts are generated from a deterministic task-stratified source subset.
  The same selected samples are used at each length; source context is truncated
  or extended through the existing normalizer, retaining the instruction suffix.
  Truncation may remove answer evidence. This measures serving cost, not quality.
- Derived prompts live under the new results directory, never `data/prepared`.
- Phase 1 prompt format v3 allocates deterministic markers using the pinned
  tokenizer so every prompt starts with a different token. Allocation covers the
  entire measured set and every warm-up wave together. Markers are selected from
  decimal and letter candidates; insufficient distinct tokens fail preparation.
  The variable body is refitted to preserve the exact token count and task
  instruction suffix. The legacy benchmark prompt builder remains unchanged.
- Saved prompt text is re-tokenized and hashed. Duplicate IDs or even a single
  shared first token across measured/warm-up requests are rejected. Actual server
  usage is still checked; no cached-token tolerance is introduced.
- Any reported cached input tokens reject the run. Missing cache usage is labeled
  protocol-only evidence, not a measured zero cache-hit rate.
- Active benchmark containers, active GPU compute processes, or a busy server port
  stop execution without killing another workload. This is a preflight check, not
  a reservation against unrelated applications starting later.
- Containers use unique names. Existing benchmark containers are never removed.
- The results directory must not exist. There is deliberately no overwrite/resume
  option in this initial workflow. Failures stop the sweep and retain artifacts.
  Rerun only the affected length/concurrency in a **new** output directory.
- Model/runtime caches remain shared as in the existing server wrapper; they may
  be populated by the engine. Existing benchmark result files are not modified.

## Safe launch sequence

Use the repository's installed virtual environment and run from the repository root.
First inspect the plan; this command performs no preparation, Docker calls, or writes:

```bash
.venv/bin/python scripts/run_context_scaling.py \
  --results-dir results/context-scaling/discovery-01 \
  --dry-run
```

Then smoke-test the smallest length at both concurrencies. Use at least four
requests so C4 actually has four requests to issue:

```bash
.venv/bin/python -u scripts/run_context_scaling.py \
  --lengths 8000 --concurrency 1,4 \
  --samples 4 --repetitions 1 \
  --results-dir results/context-scaling/smoke-8k-01 \
  --skip-image-pull
```

After checking `context_status.json`, `context_report.md`, warm-up metadata and
token usage, run all eight discovery cells in a tmux session:

```bash
.venv/bin/python -u scripts/run_context_scaling.py \
  --samples 20 --repetitions 1 \
  --results-dir results/context-scaling/discovery-01 \
  --skip-image-pull
```

Confirm the matrix in another fresh directory:

```bash
.venv/bin/python -u scripts/run_context_scaling.py \
  --samples 20 --repetitions 3 \
  --results-dir results/context-scaling/confirmation-01 \
  --skip-image-pull
```

Twenty requests per repetition supports exploratory comparison, not precise tail
estimation. For stronger request-tail evidence use `--samples 100` at selected
cells. The runner accepts up to the canonical 100 samples without duplicating them.

Optional targeted profiling, **separate** from primary latency results:

```bash
.venv/bin/python -u scripts/run_context_scaling.py \
  --lengths 64000 --concurrency 4 --samples 20 \
  --profile-nsys \
  --results-dir results/context-scaling/profile-64k-c4-01 \
  --skip-image-pull
```

`--config` can select another override file. Inspect and retain the resolved config
before comparing studies. All lengths must fit input + output within the fixed
server context ceiling; invalid options fail before workload execution.

## Evidence and interpretation

Each experiment contains a plan, pinned source hashes, code hashes and Git identity,
resolved config/environment, prepared prompts, image identity, per-cell artifacts,
CSV/Markdown reports and SHA-256 provenance. Per-cell files include client request
timings, warm-up timings, server logs/command, metrics snapshots and telemetry status.
The default telemetry records GPU, power/thermal and system resource information
where supported. Inspect availability before making memory/utilization claims;
unavailable Spark counters are not zero. GPU process overlap is not proof of GPU
occupancy. Peak memory extraction/visualization is not part of this initial report.

CSV rows retain per-repetition TTFT P50/P95, ITL P50/P95, TPOT mean, E2E P95,
output token throughput, request throughput, sample count and maximum observed ITL.
Markdown is a compact view. No averaging of percentiles is presented as a pooled
percentile, and no P99 claim is generated. ITL is based on observed SSE events, not
independently timestamped hardware tokens. Closed-loop client waiting is not a
production admission-queue measurement.

Only cells accepted after measurement, cleanup and optional CUDA trace validation
enter numeric summaries. Failed and unrun cells remain visible. Whole-process
termination (SIGKILL, power loss) may prevent final reports or cleanup; inspect the
uniquely named container and recorded plan before manually recovering. Do not
delete or overwrite the interrupted evidence directory.

The next phase adds workload mixtures, arrival-rate sweeps, cache-locality scenarios
and per-class SLOs using the capacity harness. It is intentionally outside this runner.

## Initial smoke-test correction

The initial 8K C4 warm-up reported cached tokens of 0, 8, 8, and 8. The former
32-token uniqueness check allowed a common leading header to survive. Format v2
moved hashes to the beginning but still allowed first-token collisions: w0-0 and
w0-2 both began with token 69, and one reported one cached token. Format v3 assigns
distinct first tokens using the actual tokenizer, then validates the fitted text.
This does not relax zero-cache acceptance or disable engine prefix caching.
Retain `smoke-8k-01` and `smoke-8k-02`; rerun in `smoke-8k-03`. GPU validation is
required before launching the full matrix. Do not pool different prompt formats.

Offline validation used the tokenizer JSON pinned at
`6cee5e81ee83917806bbde320786a8fb61efebee`: allocation produced 108 distinct
starting tokens, and 12 synthetic prompts at each of 8,000 and 120,000 tokens
passed exact-length and first-token checks. These checks do not substitute for
the next GPU smoke test or establish engine-reported zero cache usage.
