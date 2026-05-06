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
    --kv-cache-dtype turboquant_k8v4 \
    --max-model-len 131072
```

### KV cache layout flags

The TQ+ presets encode bit widths into the name itself. K=8bit V=4bit is the headline asymmetric layout; `*_nc` means symmetric "no centroid"; `*_rv` is the rotated-V variant (better PPL, same savings).

| `--kv-cache-dtype` | K bits | V bits | symmetric? | savings vs FP16 |
| ------------------ | ------ | ------ | ---------- | --------------- |
| `auto` (default) | 16 | 16 | n/a | — |
| `fp8`, `fp8_e4m3`, `fp8_e5m2` | 8 | 8 | yes | 50% |
| `turboquant_4bit_nc` | 4 | 4 | yes | 75% |
| `turboquant_k8v4` | 8 | 4 | no | **62.5% (headline)** |
| `turboquant_k8v3` | 8 | 3 | no | 65.6% |
| `turboquant_k4v3_nc` | 4 | 3 | no | 78.1% |
| `turboquant_k3v4_nc` | 3 | 4 | no | 78.1% |
| `turboquant_3bit_nc` | 3 | 3 | yes | 81.3% |
| any of above + `_rv` | — | — | rotated V | same savings, better PPL |

`turboquant_k8v4` is the recommended default — minimal PPL drift (+0.05–0.20 absolute) at 62.5% savings.

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
    --kv-cache-dtype turboquant_k8v4 \
    --max-num-seqs 64    # was 32 with FP16
```

## Caveats

- TurboQuant requires power-of-2 `head_dim`. Models with `head_dim=80` (Qwen3-4B) are blocked until the WHT compatibility patch lands.
- Mix-and-match with PagedAttention is supported. Mix with chunked prefill is supported. Mix with speculative decoding is **not yet** validated.
- Loading a TQ+ checkpoint and serving it with `--kv-cache-dtype fp16` works but defeats the purpose.
