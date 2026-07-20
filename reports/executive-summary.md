# Executive Summary

This project is a reproducible decision framework for long-context LLM serving on NVIDIA GB10. It compares vLLM, SGLang, and TensorRT-LLM under identical 120K-token prompts, fixed 512-token generation, cold and warm shared-prefix cache states, and matched concurrency.

The supplied preliminary warm-prefix evidence favors TensorRT-LLM for TTFT, E2E latency, and request throughput at C1/C2/C4. The result is directional: it uses one repetition per configuration. The next validation gate is three repetitions, followed by Nsight Systems profiling of the cold C4 ITL P95 anomaly.
