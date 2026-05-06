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

KV cache is the dominant memory cost at long context. TurboQuant+ asymmetric (K=q8_0, V=turbo4) shrinks it ~62% with negligible accuracy loss. The savings replicate across engines and hardware vendors. `tqkit` is the proof, the tool, and the install path.

For a 14B model at 1M tokens of context:

| layout | KV cache size | fits on MI300X 192GB? |
| ------ | ------------- | --------------------- |
| FP16 | 192 GB | no (after weights, ~28 GB free) |
| TQ+ asym (K=q8_0, V=turbo4) | 72 GB | **yes** |

You can verify the math yourself:

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
tq table --model qwen2.5-14b-instruct-1m
```

## Install

```bash
pip install tqkit
```

## Usage

```bash
tq backends                                            # autodetect installed engines
tq report --model qwen2.5-14b-instruct-1m --ctx 32K    # KV cache size for one config
tq table --model qwen2.5-14b-instruct-1m               # full layout × ctx grid
tq integrate <backend>                                 # install + serve recipe
tq bench                                               # canonical benchmark (v0.3.0)
```

Example output:

```
$ tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
[KV cache] model: Qwen/Qwen2.5-14B-Instruct-1M
[KV cache] arch: layers=48 kv_heads=8 head_dim=128
[KV cache] layout: tq+asym
[KV cache] per-token: 72.0 KB (vs 192.0 KB FP16)
[KV cache] total @ 1M ctx: 72.0 GB (vs 192.0 GB FP16, 62.5% savings)
```

## Status

**v0.2.0 — alpha**. KV math + reporter + table work. Canonical bench runner with engine bridges lands in v0.3.0.

## License

Apache 2.0.
