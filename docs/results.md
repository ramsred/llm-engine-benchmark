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

The cold C4 TensorRT-LLM ITL P95 anomaly of approximately 2.15 seconds is retained as a profiling target. It should be investigated for prefill/decode interference, scheduler contention, KV-cache pressure, queueing, and request-level outliers.
