# Fairness and Limitations

Current evidence has important limits:

- The preliminary cross-engine comparison has one valid repetition per configuration.
- Results come from one NVIDIA GB10 system, one primary model, and one headline context length.
- Scheduler and batching semantics differ between runtimes.
- Memory controls are similar but not semantically identical.
- Kernel and numerical execution paths differ.
- The TensorRT-LLM release candidate may change before final validation.
- Prefix-cache metrics can differ in precision and coverage across engines.
- Results apply to the tested workload and should not be generalized to all inference workloads.

The repository distinguishes observations, hypotheses, and profiler-backed conclusions. Raw logs, model weights, TensorRT engines, and large telemetry are intentionally excluded from source control.
