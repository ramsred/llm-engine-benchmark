# Third-party model, datasets, and containers

This repository does not redistribute model weights or benchmark datasets. The
`prepare`/`run` commands download source material directly from the upstream
projects and record immutable revisions in `experiment.lock.json`.

- NVIDIA RULER: https://github.com/NVIDIA/RULER (Apache-2.0 repository license)
- InfiniteBench: https://github.com/OpenBMB/InfiniteBench and
  https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench
- LongBench v2: https://github.com/THUDM/LongBench and
  https://huggingface.co/datasets/THUDM/LongBench-v2
- GPT-OSS-20B: https://huggingface.co/openai/gpt-oss-20b
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Triton Inference Server and TensorRT-LLM backend: https://github.com/triton-inference-server/server and https://github.com/triton-inference-server/tensorrtllm_backend

Review each upstream project's current license and terms before use. Container
images and model/dataset downloads are governed by their own terms, not this
repository's MIT license.
