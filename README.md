# tqkit

Unified toolkit for benchmarking and integrating **TurboQuant+** KV-cache compression across LLM inference engines.

## What this is

`tqkit` is a single CLI and Python package that talks to every inference engine that ships TurboQuant+ KV-cache compression:

- **llama.cpp** ([TheTom/llama.cpp@feature/turboquant-kv-cache](https://github.com/TheTom/llama.cpp/tree/feature/turboquant-kv-cache))
- **vLLM (CUDA)** ([TheTom/vllm@feature/turboquant-kv-cache](https://github.com/TheTom/vllm/tree/feature/turboquant-kv-cache))
- **vLLM (AMD ROCm)** ([TheTom/vllm@feature/turboquant-amd](https://github.com/TheTom/vllm/tree/feature/turboquant-amd))
- **MLX-Swift** ([TheTom/mlx@feature/turboquant-plus](https://github.com/TheTom/mlx/tree/feature/turboquant-plus))
- **vllm-swift** plugin

You bring the inference engine. `tqkit` autodetects what's installed, runs the canonical benchmark, and prints a reproducible KV-savings table.

## Why this exists

KV cache is the dominant memory cost at long context. TurboQuant+ shrinks it ~70% with negligible accuracy loss. The savings replicate across engines and hardware vendors. `tqkit` is the proof, the tool, and the install path.

A 14B model's KV cache at 1M tokens in FP16 is ~200 GB. With TurboQuant+ asymmetric quantization, it's ~56 GB — small enough to fit on a single MI300X.

## Install

```bash
pip install tqkit
```

## Usage

```bash
tq backends                # autodetect installed engines + versions
tq bench                   # run canonical KV-savings benchmark
tq report                  # print the most recent KV-cache layout report
tq integrate <backend>     # print install + serve recipe for one engine
```

## Status

**v0.1.0 — alpha**. Backend detection + version reporting work. Canonical bench runner and per-engine bridges land in v0.2.0.

## License

Apache 2.0.
