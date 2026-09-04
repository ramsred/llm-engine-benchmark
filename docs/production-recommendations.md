# Production Recommendations

| Requirement | Current recommendation | Reason |
| --- | --- | --- |
| Lowest warm-prefix TTFT | TensorRT-LLM | Lowest measured preliminary TTFT at C1/C2/C4 |
| Highest warm-prefix throughput | TensorRT-LLM | Highest measured preliminary request and output throughput |
| Best lightly loaded decode latency | vLLM | Candidate based on supplied preliminary C1 TPOT result; validate repeatedly |
| Flexible cache-aware serving | SGLang | Consistently competitive in the supplied comparison; validate on target traffic |
| NVIDIA-optimized production stack | TensorRT-LLM + Triton | Optimized runtime plus production serving and observability; Triton comparison is planned |
| Fast experimentation and broad compatibility | vLLM | Strong ecosystem and simpler model enablement |
| Latency-sensitive TensorRT-LLM 120K cold C4 | 2048-token prefill budget | Lowest validated P95 TTFT and P95 ITL among the repeated 8192/4096/2048 study |
| Balanced TensorRT-LLM 120K cold C4 | 4096-token prefill budget | Highest validated output throughput and lowest P95 E2E in the repeated study |

These are workload-specific recommendations, not universal engine rankings.

The prefill-budget recommendations come from three 100-request repetitions per setting. The
cross-engine recommendations remain preliminary until the complete three-repetition engine matrix
is available. Keep engine comparison controls matched by intent; do not compare a tuned
TensorRT-LLM result against untuned competing engines as if it were neutral evidence.
