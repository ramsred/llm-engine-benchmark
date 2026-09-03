# Nsight Systems Profiling Runbook

Profiling is an evidence-gathering phase, not a substitute for the neutral benchmark.

## First investigation: TensorRT-LLM cold C4

The checked-in preliminary results contain a cold C4 ITL P95 outlier. Investigate it in
two phases so profiler overhead is not confused with normal serving performance.

### Phase A: confirm the symptom without a profiler

Run three repetitions using the full 100-request population:

```bash
./bench run \
  --config config/tensorrt-llm-profile-c4.yaml \
  --results-dir results/profiling/tensorrt-llm-cold-c4-baseline \
  --samples 100 \
  --repetitions 3 \
  --overwrite
```

Confirm whether ITL P95 is consistently elevated and identify the affected requests in
`request_results.jsonl`. Do not profile until the symptom is reproducible.

### Phase B: capture a representative trace

Start with 20 requests to keep the trace manageable:

```bash
./bench run \
  --config config/tensorrt-llm-profile-c4.yaml \
  --profile-nsys \
  --overwrite
```

If the outlier does not occur in the 20-request trace, repeat with `--samples 100`.

The harness profiles the inference server inside its GPU container. On DGX Spark/GB10 it uses
Nsight's software CUDA collector (`cuda-sw`) because the default hardware collector can create a
report without CUDA activities. The profiling container receives `SYS_PTRACE`; ordinary benchmark
runs do not. The harness performs a fail-fast check for `nsys`, records the profiler version, sends
`SIGINT` during shutdown so Nsight can export cleanly, and rejects the run unless the report contains
both CUDA API and GPU-kernel summaries.

Each run directory retains:

| Artifact | Purpose |
| --- | --- |
| `profiling/server.nsys-rep` | Nsight Systems timeline |
| `profiling/nsys_version.txt` | Profiler version |
| `profiling/profile_manifest.json` | Capture completeness and trace domains |
| `profiling/cuda_summary.txt` | CUDA API and GPU-kernel summaries generated with the collector image |
| `profiling/cuda_trace_validation.json` | Machine-readable CUDA trace validation result |
| `server_command.txt` | Exact container and profiler command |
| `request_results.jsonl` | Per-request TTFT, ITL, and E2E evidence |
| `telemetry/` and `metrics_diff.json` | GPU/host and engine metrics |
| `environment/` | Driver, CUDA, image, and benchmark provenance |

The first trace includes server startup and model initialization as well as the measured
workload. Use the request results and timestamped server log to locate the measured interval;
do not interpret initialization kernels as request-serving behavior.

Open the trace in Nsight Systems UI, or generate the standard command-line summaries:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
  results/profiling/tensorrt-llm-cold-c4-trace/tensorrt_llm/cold/c4/run_01/profiling/server.nsys-rep
```

## Analysis questions

1. Does the outlier align with a long GPU kernel, a gap between launches, or scheduler queueing?
2. Are decode iterations interrupted while another request performs prefill work?
3. Does batch composition or active-sequence count change immediately before the ITL spike?
4. Is GPU compute saturated, or are there CPU/runtime launch gaps?
5. Do KV-cache allocation, eviction, or memory-pressure metrics change at the same time?

After cold C4 is explained, capture matched cold C1 and warm C4 controls and compare prefill/decode
overlap, launch behavior, and memory pressure across engines.

Do not turn an observed correlation into a root-cause claim until the trace supports it.
