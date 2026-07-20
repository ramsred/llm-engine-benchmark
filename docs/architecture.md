# Architecture

The benchmark separates workload construction, serving, measurement, and interpretation.

```mermaid
flowchart TD
    A[Canonical 120K Prompt Dataset] --> B[Neutral Benchmark Client]
    B --> C[vLLM OpenAI Server]
    B --> D[SGLang OpenAI Server]
    B --> E[TensorRT-LLM OpenAI Server]
    B --> F[Triton integration - planned]
    C --> G[Shared Metrics Pipeline]
    D --> G
    E --> G
    F --> G
    G --> H[Per-request Results]
    G --> I[Summary CSV]
    G --> J[Telemetry]
    G --> K[Dashboard and Report]
```

Each engine runs sequentially on the same NVIDIA GB10 host. The client owns request scheduling and metric formulas; engine-native benchmark CLIs are not used.
