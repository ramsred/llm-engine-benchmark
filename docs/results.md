# Results and Analysis

The checked-in headline table is a preliminary warm shared-prefix comparison generated from one accepted benchmark repetition per configuration. It is directional evidence, not a statistically validated final ranking.

Source summary: `results/summaries/combined-summary.csv`
Chart provenance: `assets/charts/provenance.json`

## Preliminary warm shared-prefix results

| Concurrency | TensorRT-LLM TTFT | SGLang TTFT | vLLM TTFT | TensorRT-LLM E2E | SGLang E2E | vLLM E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 12.88 s | 16.0 s | 22.6 s | 27.95 s | 32.5 s | 35.4 s |
| C2 | 21.76 s | 26.1 s | 36.2 s | 47.04 s | 51.9 s | 64.3 s |
| C4 | 34.41 s | 46.4 s | 59.7 s | 80.58 s | 95.8 s | 116.3 s |

TensorRT-LLM also leads the supplied preliminary request-throughput table at C1/C2/C4. The warm cache-hit ratio reported for TensorRT-LLM was 83.338%, close to the theoretical 83.33% reusable-prefix fraction.

## Observation, hypothesis, evidence

- Observation: TensorRT-LLM measured warm C4 TPOT of 87.79 ms/token.
- Hypothesis: execution planning, CUDA Graphs, batching, or kernel choice may contribute.
- Evidence required: Nsight Systems traces, runtime metrics, kernel timing, queueing data, and batch formation.
- Conclusion: TensorRT-LLM is the measured leader for this workload; the experiment does not isolate the causal mechanism.

## TensorRT-LLM cold-C4 prefill-budget study

The cold-C4 anomaly was reproduced, profiled, and tested with a causal intervention. Across three
100-request repetitions, reducing the token budget from 8192 to 2048 reduced P95 TTFT from
87.162 seconds to 63.749 seconds and P95 ITL from 2.216 seconds to 1.284 seconds. Output throughput
changed from 9.226 to 9.029 tokens per second. A 4096-token budget produced the highest measured
output throughput, 9.314 tokens per second, and the lowest measured P95 E2E, 224.406 seconds.

Nsight evidence explains the latency change. The longest long-context attention kernel fell from
474.804 milliseconds at budget 8192 to 126.315 milliseconds at budget 2048. The longest
`cudaEventSynchronize` wait fell from 5.792 seconds to 1.485 seconds. The synchronization call is
where the host waited for GPU completion; the long prefill execution window is the bottleneck.

See the [complete case study](prefill-budget-case-study.md) for repeated-run variation, limitations,
operating-point recommendations, and the reproduction command.
