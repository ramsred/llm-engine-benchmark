# Production Recommendations

| Requirement | Current recommendation | Reason |
| --- | --- | --- |
| Lowest warm-prefix TTFT | TensorRT-LLM | Lowest measured preliminary TTFT at C1/C2/C4 |
| Highest warm-prefix throughput | TensorRT-LLM | Highest measured preliminary request and output throughput |
| Best lightly loaded decode latency | vLLM | Candidate based on supplied preliminary C1 TPOT result; validate repeatedly |
| Flexible cache-aware serving | SGLang | Consistently competitive in the supplied comparison; validate on target traffic |
| NVIDIA-optimized production stack | TensorRT-LLM + Triton | Optimized runtime plus production serving and observability; Triton comparison is planned |
| Fast experimentation and broad compatibility | vLLM | Strong ecosystem and simpler model enablement |

These are workload-specific recommendations, not universal engine rankings.
