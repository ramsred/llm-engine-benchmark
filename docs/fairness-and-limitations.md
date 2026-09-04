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
- The prefill-budget study is statistically repeated but covers TensorRT-LLM cold C4 only. It is a
  scheduler-operating-point study, not a replacement for the neutral cross-engine matrix.
- Validation budgets were executed sequentially on one GB10, so run-order, thermal, and machine
  effects cannot be completely excluded. Replication on the second DGX Spark is planned.

The repository distinguishes observations, hypotheses, and profiler-backed conclusions. Raw logs, model weights, TensorRT engines, and large telemetry are intentionally excluded from source control.
