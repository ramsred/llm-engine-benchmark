# Results Artifacts

Tracked artifacts are compact and reviewable: summary CSVs, representative metadata, and report outputs. Raw request logs, server logs, telemetry, model weights, Hugging Face caches, and TensorRT engines are ignored.

The cross-engine `summaries/` files are preliminary portfolio evidence and must not be confused with a completed three-repetition matrix. The TensorRT-LLM prefill-budget summary is generated separately from its three-repetition validation and paired Nsight profiles. Full artifacts can be attached to a GitHub Release with the matching lock file, image digest, environment manifest, and benchmark commit.

Capacity experiments retain compact CSV and Markdown decisions near their run root. Raw per-request
timings, telemetry, metrics, and server logs remain ignored and belong in a checksummed release
bundle when the result is published.
