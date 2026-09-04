# TensorRT-LLM Long-Context Prefill-Budget Case Study

## Problem

TensorRT-LLM cold-prefix serving at concurrency four showed multi-second inter-token-latency
spikes for 120,000-token prompts. Multiple requests paused at the same time, pointing to shared
GPU or scheduler interference rather than an individual client or network delay.

## Method

The investigation used GPT-OSS-20B with 120,000 input tokens, 512 output tokens, FP8 KV cache,
continuous batching, and chunked prefill on one NVIDIA GB10. The exploratory sweep tested token
budgets 8192, 4096, 2048, and 1024. The final validation retained 8192, 4096, and 2048 and ran
three repetitions of 100 requests for each setting: 900 accepted requests with zero failures.

The 8192 and 2048 settings were also captured with Nsight Systems' software CUDA collector. The
client metrics and profiler traces are separate evidence: client results quantify user-visible
behavior, while CUDA runtime and kernel records explain the execution mechanism.

## Repeated validation

Values are mean ± sample standard deviation across three accepted repetitions.

| Budget | P95 TTFT | P95 ITL | Mean TPOT | P95 E2E | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 87.162 ± 2.802 s | 2.216 ± 0.010 s | 0.288 ± 0.001 s | 229.749 ± 3.802 s | 9.226 ± 0.105 |
| 4096 | 70.404 ± 0.477 s | 1.986 ± 0.023 s | 0.304 ± 0.004 s | 224.406 ± 2.139 s | 9.314 ± 0.080 |
| 2048 | 63.749 ± 2.145 s | 1.284 ± 0.032 s | 0.323 ± 0.006 s | 229.986 ± 5.594 s | 9.029 ± 0.202 |

Relative to 8192, the 2048 budget reduced P95 TTFT by 26.9% and P95 ITL by 42.1%, while output
throughput decreased by 2.1%. The 4096 budget produced the highest measured throughput and lowest
P95 E2E, making it the balanced operating point for this workload.

## Profiler evidence

| Budget | Profiled P95 ITL | Longest `cudaEventSynchronize` | Longest GPU kernel |
| ---: | ---: | ---: | ---: |
| 8192 | 1.961 s | 5.792 s | 474.804 ms |
| 2048 | 1.262 s | 1.485 s | 126.315 ms |

The longest kernels in both traces were long-context paged-KV FlashAttention kernels. Reducing the
budget by four times shortened the longest kernel by 73.4% and the longest synchronization wait by
74.4%. `cudaEventSynchronize` is the host-side observation point, not the root cause: the host was
waiting for the preceding GPU work.

The evidence supports this causal chain:

```text
larger prefill budget
  -> larger long-context prefill chunks
  -> longer attention and model-execution windows
  -> fewer opportunities to schedule active decode work
  -> higher inter-token latency
```

## Decision

- Use a 2048-token budget for latency-sensitive long-context traffic.
- Use a 4096-token budget when balancing streaming latency, completion latency, and throughput.
- Do not treat one budget as a universal default; select it from workload distributions and SLOs.
- Keep the neutral cross-engine comparison at the same matched-intent budget and report this
  TensorRT-LLM tuning experiment separately.

## Reproduction

Generate the compact evidence directly from the accepted validation runs and Nsight SQLite files:

```bash
python3 scripts/analyze_prefill_budget.py \
  --profile 8192=results/profiling/tensorrt-llm-cold-c4-cuda-sw-trace/tensorrt_llm/cold/c4/run_01 \
  --profile 2048=results/profiling/tensorrt-llm-cold-c4-budget-2048-confirmation/tensorrt_llm/cold/c4/run_01
```

Raw request logs, telemetry, and profiler reports remain outside Git. Publish them as a separately
checksummed evidence bundle with the benchmark commit, model revision, image digest, and environment
metadata.
