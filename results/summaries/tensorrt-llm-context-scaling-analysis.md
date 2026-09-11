# TensorRT-LLM Context-Scaling Analysis

- Evidence status: **complete_discovery**.
- Accepted cells: 8 / 8.
- Repetitions expected per cell: 1.
- Controlled closed-loop cold-prefix characterization; not production capacity.

## Context matrix

| Input tokens | C | Runs | TTFT P95 | ITL P95 | E2E P95 | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 1 | 1 | 0.879 s | 0.023 s | 11.966 s | 42.859 |
| 8000 | 4 | 1 | 2.661 s | 0.040 s | 23.643 s | 93.228 |
| 32000 | 1 | 1 | 5.531 s | 0.024 s | 17.522 s | 29.314 |
| 32000 | 4 | 1 | 16.974 s | 0.306 s | 52.228 s | 48.011 |
| 64000 | 1 | 1 | 16.864 s | 0.027 s | 30.146 s | 17.081 |
| 64000 | 4 | 1 | 51.807 s | 0.663 s | 120.516 s | 22.783 |
| 120000 | 1 | 1 | 50.050 s | 0.031 s | 65.361 s | 7.860 |
| 120000 | 4 | 1 | 146.148 s | 1.225 s | 305.989 s | 9.362 |

## Concurrency effect

| Input tokens | TTFT C4/C1 | ITL C4/C1 | E2E C4/C1 | Throughput C4/C1 |
| ---: | ---: | ---: | ---: | ---: |
| 8000 | 3.03× | 1.79× | 1.98× | 2.18× |
| 32000 | 3.07× | 12.48× | 2.98× | 1.64× |
| 64000 | 3.07× | 24.44× | 4.00× | 1.33× |
| 120000 | 2.92× | 39.40× | 4.68× | 1.19× |

## Interpretation guardrails

- This is complete discovery evidence, but one run per cell does not establish repeatability or confidence intervals.
- P95 values are per-run summaries; averaging repeated P95 values is not a pooled percentile.
- Concurrency ratios describe this fixed workload and do not identify a kernel-level cause.
- Use separate Nsight runs before attributing changes to attention, KV memory, or scheduling.
