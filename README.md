# Fair Multi-Backend Long-Context Performance Benchmark

A reproducible, neutral benchmark harness for comparing **SGLang**, **vLLM**, and **TensorRT-LLM**
serving `openai/gpt-oss-20b` on one NVIDIA GB10 / DGX Spark system. The project
implements the supplied benchmark design rather than calling each engine's own
benchmark generator.

This is a **serving-performance and cache-behavior benchmark**. Dataset answers
are retained for provenance, but answer quality is not scored; results must not
be presented as a model- or engine-quality ranking.

The default full experiment uses:

- exactly **120,000 input tokens** per measured request;
- exactly **512 generated tokens**, with EOS ignored and strict token-count
  rejection;
- **100 canonical samples**: 40 RULER task-family samples, 30 InfiniteBench,
  and 30 LongBench v2;
- cold unique-prefix and warm shared-prefix suites;
- concurrency **1, 2, and 4**;
- **three repetitions**, with engine order alternated and recorded;
- fresh, sequential Docker servers on the same machine;
- one asynchronous OpenAI-compatible streaming client for every backend;
- server logs, commands, image digests, environment capture, Prometheus
  snapshots, GPU/CPU telemetry, per-request timings, and report tables.

## One-command use

```bash
unzip llm-engine-benchmark.zip
cd llm-engine-benchmark
./bench run \
  --engines both \
  --modes cold,warm_shared \
  --concurrency 1,2,4 \
  --repetitions 3
```

On first launch, `./bench` creates a local `.venv`, installs the Python
environment from the exact versions in `requirements.lock`, resolves immutable
model/dataset/source revisions, downloads the
input data and tokenizer, generates both prompt suites, re-tokenizes every saved
prompt for validation, pulls the configured container images, and runs the
matrix sequentially.

Optional bootstrap/runtime variables can be placed in a local `.env` copied
from `.env.example`. The launcher accepts simple `KEY=VALUE` records without
executing the file as shell code, and `.env` is excluded from version control:

```bash
cp .env.example .env
# Edit .env if HF_TOKEN or another override is needed.
```

The host package inventory is captured as `results/environment/python_packages.txt`.


The equivalent convenience script is:

```bash
./scripts/full_benchmark.sh
```

## Recommended workflow

Check the host before downloading or launching containers:

```bash
./bench doctor
```

Run a five-request smoke test against both engines:

```bash
./scripts/smoke_test.sh --overwrite --cooldown-seconds 5
```

Then run the full accepted experiment:

```bash
./bench run --overwrite
```

Resume only already-valid matching runs after an interruption:

```bash
./bench run --resume
```

Rebuild the report without rerunning inference:

```bash
./bench report
```

## Main flags

```text
--engines both|vllm|sglang|tensorrt_llm
--modes cold,warm_shared,exact_repeat
--concurrency 1,2,4
--repetitions 3
--samples 100                 # 5 or 20 for smoke testing
--run-order alternate|sglang-first|vllm-first|random
--cooldown-seconds 60
--resume | --overwrite
--keep-going
--skip-image-pull
--no-telemetry
--dry-run
--force-prepare
--refresh-lock
```

Matched configuration can also be changed from the command line:

```text
--input-tokens 120000
--output-tokens 512
--context-length 131072
--shared-prefix-tokens 100000
--warm-groups 10
--memory-fraction 0.85

--prefill-budget 8192
--kv-cache-dtype fp8_e4m3
--vllm-image nvcr.io/nvidia/vllm:26.06-py3
--sglang-image lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
--gpus all
```

When `--samples` is smaller than 100, the harness selects a deterministic
task-stratified subset rather than taking the first rows of the manifest.


For arbitrary engine switches, use a YAML override file or append an argument:

```bash
./bench run --config my-config.yaml
./bench run --vllm-extra-arg=--some-vllm-flag
./bench run --sglang-extra-arg=--some-sglang-flag
./bench run --tensorrt-llm-extra-arg=--some-trtllm-flag
```

View the fully resolved configuration:

```bash
./bench show-config
```

## TensorRT-LLM direct backend

Direct TensorRT-LLM serving uses the same benchmark plan, prepared JSONL files,
streaming client, validation, result schema, and reports:

```bash
./bench run --engines tensorrt_llm --modes cold,warm_shared
```

Run the one-request cold smoke configuration first:

```bash
./bench run --config config/tensorrt-llm-smoke.yaml --skip-image-pull
```

For a combined three-engine comparison, reuse matching valid SGLang and vLLM
runs and add TensorRT-LLM with:

```bash
./bench run --engines sglang,vllm,tensorrt_llm \
  --modes cold,warm_shared --concurrency 1,2,4 --repetitions 3 --resume
```

This starts the configured `trtllm-serve serve` image and pins the Hugging Face
model and tokenizer to `experiment.lock.json`. A separate model/tokenizer
revision pair, an unsupported KV-cache dtype, or a server that advertises a
different model is rejected. Chunked prefill is enabled, and TensorRT-LLM's
default KV-block reuse supplies cross-request prefix caching.
The default image is NVIDIA TensorRT-LLM `1.3.0rc21`; its native `TRTLLM`
attention backend is retained without a FlashInfer override.

Triton serving is deliberately deferred; this branch supports only direct
`trtllm-serve` so an unverified model-repository path cannot be selected.

The legacy `both` shorthand remains `sglang,vllm`; use a comma-separated list
to compare the additional direct TensorRT-LLM backend.

## Host requirements

- Linux with Bash and Python 3.10 or newer.
- Docker Engine with the NVIDIA Container Toolkit.
- An NVIDIA GPU visible to `nvidia-smi`; the supplied design targets one GB10.
- Network access on the first preparation run for Hugging Face, GitHub, NGC,
  and Docker Hub.
- Enough disk for container layers, model cache, source datasets, prepared
  prompt JSONL, and result evidence. `doctor` warns below 20 GiB free, but model
  and container caches may require more.
- An authenticated `docker login nvcr.io` when the selected NGC image requires
  it.

`pidstat` is optional. When unavailable, the built-in `psutil` telemetry still
captures CPU, memory, swap, and load data.

## Automatic data acquisition and pinning

The first `prepare` or non-dry `run` command creates `experiment.lock.json`.
It records:

- the exact GPT-OSS model commit;
- the exact tokenizer commit;
- the exact InfiniteBench dataset commit;
- the exact LongBench v2 dataset commit, including any repository redirect;
- the exact NVIDIA RULER Git commit;
- Docker image IDs/digests captured with the run evidence; servers are launched by the resolved immutable local image ID rather than the mutable tag.

Data sources are not embedded in this repository. They are downloaded at run
time:

- InfiniteBench from `xinrongzhang2022/InfiniteBench` using the official task
  splits;
- LongBench v2 from `THUDM/LongBench-v2` using its official `train` split;
- the pinned NVIDIA RULER repository for source provenance.

RULER is a configurable synthetic benchmark rather than a static 40-row file.
This project therefore uses a deterministic, neutral built-in generator for the
four task families required by the design—needle retrieval, variable
tracking/multi-hop, aggregation/counting, and QA—while recording and fetching
the upstream RULER revision. It deliberately does not use an engine-specific
RULER prompt template.

To prepare and validate without starting an inference server:

```bash
./bench prepare
```

To update all upstream revisions intentionally:

```bash
./bench prepare --refresh-lock --force-prepare
```

A refresh changes experiment identity. Do not mix results made from different
lock files. Reusing a lock after changing a model, tokenizer, dataset repository,
or requested revision is rejected until `--refresh-lock` is supplied.

## Canonical allocation

The manifest contains exactly 100 fixed source records.

| Source | Category | Count |
|---|---|---:|
| RULER | Needle/retrieval | 10 |
| RULER | Variable tracking or multi-hop | 10 |
| RULER | Aggregation/counting | 10 |
| RULER | QA | 10 |
| InfiniteBench | Fake-book QA / MC | 12 |
| InfiniteBench | Dialogue | 4 |
| InfiniteBench | Code debugging | 4 |
| InfiniteBench | Pass-key / number / KV retrieval | 10 |
| LongBench v2 | Single- and multi-document QA | 10 |
| LongBench v2 | Long ICL and dialogue | 10 |
| LongBench v2 | Code repository and structured data | 10 |

Selections are deterministic from the configured seed and upstream immutable
IDs. The canonical records are stored in `data/canonical/manifest.jsonl`.

## Exact prompt normalization

Preparation uses the pinned GPT-OSS tokenizer with
`add_special_tokens=False`. For every measured record it:

1. keeps context before the instruction/question;
2. reserves the instruction as a fixed suffix, enforces the configured
   1,024-token instruction budget, and truncates only context;
3. extends short sources with deterministic, lexically varied controlled
   distractor paragraphs rather than one repeated token;
4. fits a tokenizer-prefix of the context to the target;
5. saves the resulting prompt string;
6. reloads the JSONL and re-tokenizes the saved string;
7. rejects preparation unless every prompt has exactly 120,000 tokens.

The cold suite starts each prompt with a long sample-specific prelude and rejects
any duplicate first-256-token hash.

The warm suite builds ten groups. Each group has:

- one exactly 100,000-token shared prefix;
- ten measured requests with an exactly 20,000-token unique suffix;
- one sequential prefix warm-up request;
- a SHA-256 hash of the shared token sequence.

The validator checks that the first 100,000 token IDs of every member are
identical to the group's saved prefix—not merely similar text.

## Server parity

The default server intent is matched as follows:

| Property | vLLM | SGLang |
|---|---|---|
| Context | `--max-model-len 131072` | `--context-length 131072` |
| Prefix cache | `--enable-prefix-caching` | RadixAttention/default radix cache |
| KV cache | `--kv-cache-dtype fp8_e4m3` | `--kv-cache-dtype fp8_e4m3` |
| Memory fraction | `--gpu-memory-utilization 0.85` | `--mem-fraction-static 0.85` |
| Prefill intent | `--max-num-batched-tokens 8192` | `--chunked-prefill-size 8192` |
| Chunked prefill | enabled | enabled by configured chunk size |
| Weight quantization | `mxfp4` | `mxfp4` |
| CUDA graphs | default/enabled | default/enabled |
| Sampling defaults | `--generation-config vllm` plus explicit request values | `--sampling-defaults openai` plus explicit request values |
| Usage/cache evidence | streamed usage requested | streamed usage plus `--enable-cache-report` |
| Parallelism | one GPU | one GPU |

The scheduler controls are matched by intent but are not mechanically identical.
The generated report repeats this limitation and does not claim they are the
same implementation.

The vLLM launch also retains the GPT-OSS-specific sliding-window KV dtype skip
setting from the supplied reference script. It can be removed in a YAML
override when testing an image that does not expose that option. vLLM model-side
generation defaults are disabled with `--generation-config vllm`, while SGLang
uses `--sampling-defaults openai`; the neutral request still pins every common
sampling control explicitly. SGLang metrics and cache reporting are enabled by
default for cache evidence.

## Cache-state protocol

### Cold unique-prefix

For every engine, concurrency, and repetition, the harness:

1. removes any old benchmark container;
2. starts a fresh server;
3. waits for `/v1/models` readiness;
4. sends one unrelated prompt validated at exactly 32 tokens to initialize runtime paths;
5. sends no benchmark prompt before measurement;
6. captures a pre-run metrics snapshot;
7. starts telemetry;
8. measures the selected cold prompts;
9. captures post-run metrics, logs, inspect data, and validation evidence;
10. stops and removes the server.

The unrelated 32-token warm-up intentionally avoids every benchmark prefix.
It does not attempt to precompile long-context kernels, so compilation or
autotuning triggered by the first 120K-token measured request is part of the
observed cold behavior for both engines.

### Warm shared-prefix

The harness starts a fresh server, performs the unrelated runtime warm-up, then
sends exactly one prefix warm-up for each group represented in the measured
sample. Prefix warm-ups are sequential and finish before telemetry and measured
requests begin. The server is not restarted or flushed between warming and
measurement.

### Exact repeat

`--modes exact_repeat` is optional. It starts a fresh server, sends every selected
complete cold prompt once with a one-token completion to populate cache, and
then measures an identical second pass. This is treated as a cache-capacity and
eviction test, not a simple guaranteed-hit test.

## Neutral client and validation

Both engines receive raw POST requests to `/v1/completions` with the same saved
prompt and common body:

```json
{
  "model": "openai/gpt-oss-20b",
  "prompt": "<exact saved prompt>",
  "max_tokens": 512,
  "temperature": 0,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": true,
  "stream_options": {"include_usage": true},
  "n": 1,
  "echo": false,
  "ignore_eos": true,
  "add_special_tokens": false,
  "skip_special_tokens": false
}
```

Only the base URL and configurable engine-extension mapping differ. No engine's
own benchmark CLI generates or schedules requests.

The client records:

- TTFT: first non-empty streamed token event minus request start;
- E2E latency;
- token event timestamps and ITLs;
- TPOT: `(completion - first token) / (output tokens - 1)`;
- request throughput;
- output-token throughput;
- logical input-token throughput;
- server-reported prompt, completion, and cached-prompt token usage when exposed;
- observed uncached-token throughput and cache-hit ratio, with usage-report coverage;
- failures, HTTP errors, finish reasons, and exact output-token counts.

When one SSE event contains multiple tokens, those tokens share the observed
event timestamp. The client does not fabricate interpolated token times.

A run is rejected when any request fails, the server is no longer running after
measurement, a server-reported prompt count disagrees with the canonical count,
or the generated token total differs from `sample_count × output_tokens`. A full
100-request configuration must therefore contain exactly **51,200 output
tokens**. Streamed server usage is authoritative for completion-token totals;
re-tokenizing decoded text is retained only as a diagnostic. With the default
`project.require_server_token_usage: true`, a server that omits authoritative
completion-token usage causes the run to be rejected.

Warm logical input throughput is explicitly labeled as cache-assisted prompt
acceptance. When complete prompt/cache usage details are available, the client
reports observed uncached prompt tokens and cache-hit ratio. Otherwise it stores
the known 20K-per-request unique-suffix lower bound separately and relies on
`metrics_diff.json` and `server.log` as additional cache evidence.

## Result layout

```text
results/
├── active_experiment.json
├── run_plan.json
├── environment/
│   ├── experiment_lock.json
│   ├── docker_images.json
│   ├── gpu.txt
│   ├── uname.txt
│   └── ...
├── sglang/
│   ├── cold/c1/run_01/
│   ├── cold/c2/run_01/
│   ├── cold/c4/run_01/
│   └── warm_shared/c1/run_01/ ...
├── vllm/
│   └── ...
└── report/
    ├── report.md
    ├── runs.csv
    ├── summary.csv
    ├── cold.csv
    ├── warm_shared.csv
    ├── exact_repeat.csv
    └── by_source.csv
```

Every run directory contains, when available:

```text
client_results.json
request_timings.jsonl
requests.reference.json
run_metadata.json
server_command.txt
image_digest.txt
docker_inspect.json
engine_runtime.json
server.log
metrics_before.prom

metrics_after.prom
metrics_diff.json
telemetry.csv
host_telemetry.jsonl
cpu_memory.txt
warmup_results.json
```

`report/report_status.json` and the report header mark the matrix `COMPLETE`
only when every run listed in `active_experiment.json` exists and is accepted.
Partial reports remain usable for diagnosis but are prominently labeled incomplete.

Prepared prompts are referenced by immutable checksum instead of copied into all
36 run directories.

## Statistical reporting

`./bench report` keeps cold and warm observations separate. For each engine,
mode, and concurrency it reports:

- the median run duration and observed min/max/CV;
- median-of-repetition TTFT P50/P95, TPOT P50, ITL P95, and E2E P50/P95;
- median request, logical-input, computed-input, and output throughput;
- median cache-report coverage, cached-token total, and cache-hit ratio when exposed;
- output-throughput min/max/CV;
- per-source RULER, InfiniteBench, and LongBench v2 latency breakdowns.

All raw request-level observations remain available for alternative analyses.
`active_experiment.json` scopes report generation so stale runs from an older
image digest, sample count, mode set, or matrix are not silently mixed into the
current report.

## Configuration file

`config/default.yaml` is the source of truth. A custom file is deep-merged over
it:

```yaml
project:
  model_revision: "<commit SHA>"
  cooldown_seconds: 30

engines:
  vllm:
    image: "your-vllm-image:tag"
    extra_args:
      - --some-flag
  sglang:
    image: "your-sglang-image:tag"
```

Launch it with:

```bash
./bench run --config custom.yaml
```

## Reproducibility rules

- Keep `experiment.lock.json`, `active_experiment.json`, and `run_plan.json` with the result bundle.
- Do not run SGLang and vLLM simultaneously.
- Do not compare runs made with different prepared prompt checksums.
- Do not merge cold and warm distributions.
- Preserve rejected runs; they are useful diagnostic evidence but are excluded
  from accepted summary tables.
- Record any manual clock, power, thermal, Docker daemon, or operating-system
  changes outside the harness.
- Prefer `--resume` after interruption. Use `--overwrite` only when deliberately
  replacing evidence.

## Troubleshooting

**NGC pull fails**  
Run `docker login nvcr.io`, confirm the requested tag exists for your account,
or supply `--vllm-image`. `--skip-image-pull` uses an already-present local
image and still records its local ID/digest.

**A server flag is unsupported**  
Container releases change. Pin a compatible image and override the flag list in
YAML. The failed run retains `server.log` and `server_command.txt`.

**Output is shorter than 512 tokens**  
The run is intentionally rejected. Inspect whether the selected server release
accepts the `ignore_eos` OpenAI extension. Adjust only the engine-specific
`request_extra` mapping while preserving identical effective output semantics.

**Warm prefixes do not hit cache**  
Inspect `metrics_diff.json` and `server.log`. The dataset validator already
proves token-ID equality, so remaining causes include capacity, eviction,
engine block alignment, scheduler behavior, or version-specific cache policy.

**Preparation is slow**  
Exact 120K construction and validation intentionally tokenize tens of millions
of tokens. Completed files are signature-cached. Avoid `--force-prepare` unless
model/tokenizer/data settings changed.

**Port conflict**  
Use `--vllm-port` and `--sglang-port`, or stop the process/container occupying
8000 or 30000. `doctor` checks both ports.

## Local verification included with the project

The source ZIP contains unit tests for configuration gates,
exact fitting with a controlled tokenizer, shared-prefix preservation, dataset
classification, neutral-client token validation, cache-aware reporting, server
argument parity, run-matrix ordering, percentiles, deterministic hashing, and
Prometheus metric diffs:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These tests do not replace a GPU/Docker smoke run; they validate the portable
control logic.
