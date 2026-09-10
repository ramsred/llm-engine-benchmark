# TensorRT-LLM Production Capacity Analysis

## Repeated SLA validation

| Offered RPS | Status | Runs | Achieved RPS | Success | Reject | TTFT P95 | ITL P95 | E2E P95 | Queue P95 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.014000 | **PASS** | 3 | 0.014016 | 1.000 | 0.000 | 48.326 s | 0.031 s | 63.682 s | 0.000 s |
| 0.016000 | **PASS** | 3 | 0.015993 | 1.000 | 0.000 | 50.123 s | 0.869 s | 114.171 s | 0.000 s |
| 0.018000 | **UNSTABLE** | 3 | 0.017458 | 0.973 | 0.027 | 51.998 s | 1.296 s | 226.034 s | 18.456 s |
| 0.020000 | **FAIL** | 3 | 0.017361 | 0.873 | 0.127 | 59.788 s | 1.266 s | 227.892 s | 25.740 s |

## Decision

- Highest validated passing load: **0.016000 RPS**.
- Lowest validated failing load: **0.020000 RPS**.
- Validated transition interval: **(0.016000, 0.020000) RPS**.
- Unstable tested rates inside that interval: **0.018000 RPS**.
- Recommended validated operating point: **0.014000 RPS**.
- This is a workload-specific operating envelope, not a universal engine limit.
- Validate burst recovery and N+1 failover before production deployment.
