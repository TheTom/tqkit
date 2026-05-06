# Integrating TurboQuant+ into vLLM (AMD ROCm)

Use TurboQuant+ KV-cache compression with vLLM on AMD MI300X / MI250X / Radeon W7900.

This is **the only production TurboQuant implementation for AMD GPUs**. NVIDIA has its own; Apple has MLX-Swift; AMD users had nothing until this.

## Install — Docker (recommended)

```bash
docker pull thetom/vllm-turboquant:rocm-7.2

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  thetom/vllm-turboquant:rocm-7.2 \
    --model Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype tq_asym
```

The image defaults to `tq_asym` if no `--kv-cache-dtype` is passed. Override with `fp16` for an apples-to-apples baseline.

## Install — from source

Requires ROCm 7.2 (or 6.x with patches), Python 3.10+, and a working `hipblaslt`:

```bash
git clone -b feature/turboquant-amd https://github.com/TheTom/vllm
cd vllm
pip install -e .
```

The AMD-specific kernels live in `vllm/attention/backends/rocm_*` and use ROCm's MFMA matrix instructions for KV-cache scoring.

## Serve a model

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype tq_asym \
    --max-model-len 131072
```

## The 1M-context-on-MI300X demo

A 14B model at 1M tokens of context:

| layout | KV cache | fits on MI300X (192 GB)? |
| ------ | -------- | ------------------------ |
| FP16 | 192 GB | no (no headroom for weights) |
| TQ+ asym | **72 GB** | **yes** (~92 GB free for weights + workspace) |

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  thetom/vllm-turboquant:rocm-7.2 \
    --model Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype tq_asym \
    --max-model-len 1048576
```

This is the workload pinned in `tqkit/canonical_bench.yml` under `killer_demo`.

## Verify the savings

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
# → 72 GB KV cache vs 192 GB FP16 (62.5% savings)
```

## Hardware

| GPU | gfx | tested | notes |
| --- | --- | ------ | ----- |
| MI300X | gfx942 | yes | 192 GB HBM3, primary target |
| MI250X | gfx90a | not yet | should work with rebuild |
| Radeon W7900 | gfx1100 | not yet | RDNA3, untested |

## Caveats

- ROCm 7.2 only currently. ROCm 6.x backport is on the roadmap.
- MFMA-friendly scoring kernels are MI300X-tuned; performance on MI250X/W7900 may be suboptimal until per-arch tuning is added.
- The `tq_asym` path on hybrid (Mamba+Attention) models is not yet validated. Use `tq_sym` for Qwen3-Next family.
