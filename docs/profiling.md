# Profiling Plan

Profiling is an evidence-gathering phase, not a substitute for the neutral benchmark.

1. Repeat the cold C4 configuration and confirm the ITL P95 outlier.
2. Capture Nsight Systems traces for representative C1 and C4 cold/warm runs.
3. Correlate request timestamps with scheduler batches, GPU kernels, KV-cache allocation, and queueing.
4. Compare prefill/decode overlap and kernel launch behavior across engines.
5. Record profiler version, capture command, image digest, driver, and CUDA versions beside the trace.

Do not turn an observed correlation into a root-cause claim until the trace supports it.
