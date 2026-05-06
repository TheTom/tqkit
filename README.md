# tqkit

Unified toolkit for benchmarking and integrating **TurboQuant+** KV-cache compression across LLM inference engines.

## What this is

`tqkit` is a single CLI and Python package that talks to every inference engine that ships TurboQuant+ KV-cache compression:

- **llama.cpp** ([TheTom/llama.cpp@feature/turboquant-kv-cache](https://github.com/TheTom/llama.cpp/tree/feature/turboquant-kv-cache))
- **vLLM (CUDA)** ([TheTom/vllm@feature/turboquant-kv-cache](https://github.com/TheTom/vllm/tree/feature/turboquant-kv-cache))
- **vLLM (AMD ROCm)** ([TheTom/vllm@feature/turboquant-amd-noautotune](https://github.com/TheTom/vllm/tree/feature/turboquant-amd-noautotune))
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

## Integration recipes

One-page docs for plugging TurboQuant+ into each supported backend live under [`docs/integrate/`](docs/integrate/).

- [llama.cpp](docs/integrate/llama-cpp.md) — NVIDIA, Apple, AMD, CPU
- [vLLM (NVIDIA CUDA)](docs/integrate/vllm-cuda.md) — A100, H100, RTX 4090
- [vLLM (AMD ROCm)](docs/integrate/vllm-amd.md) — MI300X (the only TQ+ port for AMD anywhere)
- [MLX-Swift](docs/integrate/mlx-swift.md) — Apple Silicon Macs + iPhone
- [vllm-swift](docs/integrate/vllm-swift.md) — Apple Silicon OpenAI-API server

## Docker (AMD ROCm)

```bash
docker pull thetom/vllm-turboquant:rocm-7.2
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p 8000:8000 thetom/vllm-turboquant:rocm-7.2 \
        --model Qwen/Qwen2.5-14B-Instruct-1M --kv-cache-dtype turboquant_k8v4
```

See [`docker/README.md`](docker/README.md) for build details.

## Status

**v0.3.0 — alpha**. Shipping today:

- KV math + `tq report` + `tq table`
- Pinned `canonical_bench.yml` + `tq config`
- Engine bridges (`tq bench`) for llama.cpp, vLLM (CUDA + AMD), MLX-Swift, vllm-swift
- Integration recipes for all 5 backends (`docs/integrate/`)
- Docker scaffold for AMD ROCm (`docker/Dockerfile.vllm-amd`)
- Models supported in `tq report`: Qwen2.5 7B/14B/32B, Qwen3-8B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3-Next-80B-A3B, Llama-3.1 8B/70B, Mistral-7B
- 39 tests, 92% line coverage, ≥85% gate enforced

See [`CHANGELOG.md`](CHANGELOG.md) for the full version history.

## License

Apache 2.0.
