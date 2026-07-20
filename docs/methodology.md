# Methodology and Fairness Contract

The headline workload is GPT-OSS-20B with exactly 120,000 input tokens and 512 generated tokens. The default matrix uses cold unique-prefix and warm shared-prefix modes, concurrency 1/2/4, 100 samples, and three repetitions.

| Dimension | Control |
| --- | --- |
| Hardware | Same NVIDIA GB10 system |
| Model | Same GPT-OSS-20B revision |
| Tokenizer | Pinned tokenizer revision |
| Prompt length | Exactly 120,000 tokens |
| Output length | Exactly 512 tokens; rejected otherwise |
| Samples | 100 per configuration |
| Client | One neutral async OpenAI-compatible streaming client |
| Metrics | Identical client-side formulas |
| Cache protocol | Fresh process for cold; sequential prefix warm-up for warm |
| Memory target | 0.80 fraction, aligned by engine-specific control |

TTFT is the request start to first non-empty token event. TPOT is decode time divided by generated tokens after the first token. ITL is the interval between token events. E2E is request start to final event. Reports retain P50/P95, repetition spread, cache coverage, and telemetry evidence.

Scheduler semantics, batching, kernel selection, cache block size, and allocation policies differ across engines. Configuration values are matched by intent, not claimed to be mechanically identical.
