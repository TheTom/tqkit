# Integrating TurboQuant+ into vLLM (NVIDIA CUDA)

Use TurboQuant+ KV-cache compression with vLLM on NVIDIA GPUs.

## Install

```bash
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/vllm
cd vllm
pip install -e .
```

Tested with CUDA 12.4+ on A100, H100, RTX 4090.

## Serve a model

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype tq_asym \
    --max-model-len 131072
```

### KV cache layout flags

| `--kv-cache-dtype` | layout | savings vs FP16 |
| ------------------ | ------ | --------------- |
| `auto` / `fp16` | FP16 baseline | — |
| `fp8` | FP8 (E4M3 or E5M2) | 50% |
| `tq_sym` | TQ+ symmetric (K=q8_0+WHT, V=q8_0+WHT) | 50% |
| `tq_asym` | TQ+ asymmetric (K=q8_0+WHT, V=turbo4) | 62.5% |

`tq_asym` is the headline TurboQuant+ layout. Minimal PPL drift at the highest savings.

## Verify the savings

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 32K --layout tq+asym
```

## Memory math: how big a context can I serve?

Approximate KV cache footprint per request at FP16 vs TQ+ asym:

| GPU VRAM (after weights) | model | FP16 max ctx | TQ+ asym max ctx |
| ------------------------ | ----- | ------------ | ----------------- |
| 24 GB (RTX 4090, 14B Q4) | Qwen2.5-14B | ~110K | ~280K |
| 80 GB (A100, 14B FP16) | Qwen2.5-14B | ~270K | ~720K |
| 80 GB (H100, 14B FP16) | Qwen2.5-14B | ~270K | ~720K |

Numbers are rough — actual fit depends on workspace, kernel scratch, batch size.

## Hybrid serving (2× throughput at the same VRAM)

The simplest production win: keep your model the same, halve the KV cache, run 2× the concurrent requests.

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype tq_asym \
    --max-num-seqs 64    # was 32 with FP16
```

## Caveats

- TurboQuant requires power-of-2 `head_dim`. Models with `head_dim=80` (Qwen3-4B) are blocked until the WHT compatibility patch lands.
- Mix-and-match with PagedAttention is supported. Mix with chunked prefill is supported. Mix with speculative decoding is **not yet** validated.
- Loading a TQ+ checkpoint and serving it with `--kv-cache-dtype fp16` works but defeats the purpose.
