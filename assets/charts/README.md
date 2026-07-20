## Portfolio charts

> Preliminary results: one accepted repetition per configuration.  
> Workload: GPT-OSS-20B, exactly 120,000 input tokens, fixed 512-token output, warm shared-prefix, C1/C2/C4.

### Time to first token
![Warm shared-prefix TTFT](../01_warm_shared_ttft.png)

### Decode latency
![Warm shared-prefix TPOT](../02_warm_shared_tpot.png)

### End-to-end latency
![Warm shared-prefix E2E](../03_warm_shared_e2e.png)

### Request throughput
![Warm shared-prefix request throughput](../04_warm_shared_request_throughput.png)

### Latency-throughput frontier
![Latency-throughput frontier](../05_latency_throughput_frontier.png)

### Verified prefix-cache benefit
![TensorRT-LLM cache TTFT reduction](../06_tensorrt_cache_ttft_reduction.png)
