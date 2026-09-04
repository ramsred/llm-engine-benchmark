# TensorRT-LLM Prefill-Budget Analysis

## Repeated validation

| Budget | Valid runs | Requests | TTFT P95 | ITL P95 | TPOT mean | E2E P95 | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 3 | 300 | 87.162 ± 2.802 s | 2.216 ± 0.010 s | 0.288 ± 0.001 s | 229.749 ± 3.802 s | 9.226 ± 0.105 |
| 4096 | 3 | 300 | 70.404 ± 0.477 s | 1.986 ± 0.023 s | 0.304 ± 0.004 s | 224.406 ± 2.139 s | 9.314 ± 0.080 |
| 2048 | 3 | 300 | 63.749 ± 2.145 s | 1.284 ± 0.032 s | 0.323 ± 0.006 s | 229.986 ± 5.594 s | 9.029 ± 0.202 |

## Profiler evidence

| Budget | Client ITL P95 | Longest CUDA synchronization | Longest GPU kernel |
| ---: | ---: | ---: | ---: |
| 8192 | 1.961 s | 5.792 s | 474.804 ms |
| 2048 | 1.262 s | 1.485 s | 126.315 ms |

## Decision

- Latency-oriented operating point: **2048 tokens**.
- Balanced operating point: **4096 tokens**.
- Relative to 8192, the latency-oriented setting changed P95 ITL by -42.0% and output throughput by -2.1%.
- `cudaEventSynchronize` records where the host waited; the preceding GPU work is the causal bottleneck.
- Conclusions are workload-specific and do not establish a universal engine default.
