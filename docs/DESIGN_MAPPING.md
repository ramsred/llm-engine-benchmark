# Performance-benchmark design-to-implementation mapping

| Design control | Implementation |
|---|---|
| Identical prompts | `data/prepared/*.jsonl` is generated once; both engines receive the same file and checksum. |
| Identical tokenizer | `experiment.lock.json` pins the GPT-OSS tokenizer commit; `validate` re-tokenizes every saved prompt with `add_special_tokens=False`. |
| Exactly 120K input | `normalize.fit_variable_segment`, the 1,024-token instruction reserve, and full-file validation enforce exactly 120,000 tokens. |
| Exactly 512 output | The common OpenAI request forces `max_tokens`, deterministic sampling, and `ignore_eos`; authoritative streamed server usage is required and every mismatch rejects the run. |
| No hidden special tokens | Both request adapters set `add_special_tokens=false` and preserve decoded special-token text for neutral re-tokenization diagnostics. |
| 40/30/30 suite | `datasets.py` enforces exact source and category counts without cross-category fallback. |
| Performance-only scope | Expected answers are retained as provenance, but the harness explicitly does not score or claim answer quality. |
| Representative partial runs | Sample limits below 100 use a deterministic task-stratified selection rather than the first manifest rows. |
| Cold cache | Fresh Docker container per configuration and one unrelated, exactly 32-token runtime warm-up. |
| Warm shared prefix | Ten exact 100K-token prefixes, sequential warm-up, and a 20K-token unique measured suffix. |
| Neutral client | `client.py` sends the same async streaming request schedule to both endpoints. |
| Concurrency 1/2/4 | One client semaphore and the same saved request order. |
| Three repetitions | `orchestrator.py` builds 36 default runs. |
| Alternating order | Repetition 1 starts SGLang first, repetition 2 vLLM first, and later repetitions are seeded/randomized and recorded. |
| Matched sampling intent | vLLM uses `--generation-config vllm`; SGLang uses `--sampling-defaults openai`; common request values are explicit. |
| Cache evidence | Streamed usage/cache details are recorded when exposed, SGLang cache reporting is enabled, and Prometheus/log evidence is preserved. |
| Telemetry | `nvidia-smi dmon`, `pidstat`, and `psutil` are captured where available. |
| Raw evidence | Per-run logs, Docker inspect, image digest, runtime versions, commands, metrics snapshots, timings, and metadata. |
| Statistical treatment | Separate cold/warm/exact-repeat CSVs, repetition medians, min/max/CV, cache coverage/hit ratio, and per-source tables. |
| Known scheduler limitation | The report states that SGLang chunked-prefill size and vLLM batched-token budget are matched by intent, not mechanics. |

## Additional reproducibility guards

- Existing experiment locks are checked against the configured model, tokenizer, dataset repositories, and requested revisions; a mismatch requires `--refresh-lock`.
- Docker tags are resolved once before the matrix and each server is launched by the immutable local image ID captured in the evidence.
- Host Python packages are installed from exact versions in `requirements.lock` and `pip freeze --all` is captured with environment evidence.
- The default SGLang image is pinned by multi-architecture OCI digest rather than a mutable tag.
- `active_experiment.json` prevents generated reports from mixing runs made with another image digest or matrix scope.
