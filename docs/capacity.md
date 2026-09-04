# SLA-Constrained Production Capacity

## Question answered

The capacity workflow finds the highest offered request rate that a serving configuration can
sustain while meeting explicit latency, success, rejection, and queueing objectives. This is
different from the fixed-concurrency benchmark: arrivals occur on an independent clock even while
the server is busy.

Each cold load point starts a fresh server. This prevents an earlier rate from warming cache state
for a later rate and keeps the comparison reproducible.

## Initial 120K latency profile

The supplied configuration uses the profiler-validated 2048-token TensorRT-LLM prefill budget.
Initial SLA defaults are deliberately specific to 120,000-token cold prompts:

| Objective | Default |
| --- | ---: |
| P95 TTFT | 70 seconds |
| P95 ITL | 1.5 seconds |
| P95 E2E | 250 seconds |
| P95 admission queue time | 10 seconds |
| Minimum scheduled-request success fraction | 0.99 |
| Maximum rejection fraction | 0.01 |

These are experimental acceptance criteria, not universal chatbot SLOs. Shorter contexts require
separate, tighter objectives.

## Discovery run

Start with five load points and 20 requests per point:

```bash
./bench capacity \
  --config config/tensorrt-llm-capacity-120k.yaml \
  --rates 0.008,0.012,0.016,0.020,0.024 \
  --requests 20 \
  --repetitions 1 \
  --max-in-flight 4 \
  --queue-limit 8 \
  --queue-timeout-seconds 30 \
  --results-dir results/capacity/tensorrt-llm-120k-discovery \
  --skip-image-pull
```

The constant-arrival default makes configuration comparisons reproducible. Use
`--arrival-pattern poisson` later to model independently arriving production traffic.

## Boundary confirmation

After discovery identifies the likely saturation knee, retain one point below it, the candidate
boundary, and one point above it. Run 100 requests and three repetitions. Even repetitions reverse
the rate order to reduce run-order and thermal bias.

```bash
./bench capacity \
  --config config/tensorrt-llm-capacity-120k.yaml \
  --rates 0.012,0.016,0.020 \
  --requests 100 \
  --repetitions 3 \
  --results-dir results/capacity/tensorrt-llm-120k-validation \
  --skip-image-pull
```

## Admission semantics

- `--max-in-flight` limits requests actively sent to the inference server.
- `--queue-limit` bounds client-side admission waiting. Zero means reject immediately when every
  in-flight slot is occupied.
- `--queue-timeout-seconds` rejects a queued request that cannot obtain a slot in time.
- Queue-full and queue-timeout outcomes remain measured production results; they are not discarded.
- The requested arrival schedule continues independently of completions, preserving open-loop load.

The load generator records scheduled and actual arrival times. A run is invalid when P95 dispatch
lag exceeds `--max-arrival-lag-seconds`, because that would mean the client failed to generate the
requested traffic accurately.

## Capacity decision

A rate passes only when all repetitions are valid and satisfy every SLA check. The report selects
the highest passing rate as the measured boundary and recommends 75% of that rate as the initial
operating point with headroom.

Generated artifacts include:

- `capacity_run_plan.json`: offered load, admission policy, SLA and run order;
- `capacity_request_timings.jsonl`: arrival, admission, streaming and completion evidence;
- `capacity_results.json`: one load point's measurements and SLA checks;
- `capacity_summary.csv`: compact row for every rate and repetition;
- `capacity_report.md`: capacity boundary and 25% headroom recommendation;
- server logs, metrics, environment capture and host/GPU telemetry.

The first implementation establishes single-worker steady-load capacity. Burst recovery,
multi-replica routing, worker failure, N+1 sizing and Kubernetes admission are subsequent production
experiments and must not be claimed from this result.
