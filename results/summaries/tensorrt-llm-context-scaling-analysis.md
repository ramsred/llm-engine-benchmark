# TensorRT-LLM Context-Scaling Analysis

- Evidence status: **complete_repeated_validation**.
- Accepted cells: 24 / 24.
- Repetitions expected per cell: 3.
- Controlled closed-loop cold-prefix characterization; not production capacity.

## Context matrix

| Input tokens | C | Runs | TTFT P95 | ITL P95 | E2E P95 | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 1 | 3 | 0.859 ± 0.002 s | 0.023 ± 0.000 s | 11.948 ± 0.015 s | 42.946 ± 0.069 |
| 8000 | 4 | 3 | 2.621 ± 0.022 s | 0.040 ± 0.000 s | 23.160 ± 0.128 s | 93.854 ± 0.299 |
| 32000 | 1 | 3 | 5.597 ± 0.034 s | 0.025 ± 0.000 s | 17.643 ± 0.057 s | 29.258 ± 0.158 |
| 32000 | 4 | 3 | 17.249 ± 0.081 s | 0.310 ± 0.002 s | 52.474 ± 0.101 s | 48.122 ± 0.139 |
| 64000 | 1 | 3 | 16.495 ± 0.177 s | 0.027 ± 0.000 s | 29.742 ± 0.174 s | 17.312 ± 0.094 |
| 64000 | 4 | 3 | 50.886 ± 0.592 s | 0.660 ± 0.005 s | 119.058 ± 1.315 s | 23.002 ± 0.194 |
| 120000 | 1 | 3 | 49.667 ± 1.456 s | 0.031 ± 0.000 s | 64.931 ± 1.430 s | 7.922 ± 0.174 |
| 120000 | 4 | 3 | 150.626 ± 2.313 s | 1.261 ± 0.017 s | 315.419 ± 4.691 s | 9.123 ± 0.129 |

## Concurrency effect

| Input tokens | TTFT C4/C1 | ITL C4/C1 | E2E C4/C1 | Throughput C4/C1 |
| ---: | ---: | ---: | ---: | ---: |
| 8000 | 3.05× | 1.75× | 1.94× | 2.19× |
| 32000 | 3.08× | 12.61× | 2.97× | 1.64× |
| 64000 | 3.08× | 24.42× | 4.00× | 1.33× |
| 120000 | 3.03× | 40.64× | 4.86× | 1.15× |

## Interpretation guardrails

- P95 values are per-run summaries; averaging repeated P95 values is not a pooled percentile.
- Concurrency ratios describe this fixed workload and do not identify a kernel-level cause.
- Use separate Nsight runs before attributing changes to attention, KV memory, or scheduling.
