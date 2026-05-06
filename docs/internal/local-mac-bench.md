# Local Mac bench — initial KV measurements

Hardware: Apple M5 Max, 128 GB unified memory
Date: 2026-05-06

## Engines tested

| engine | binary version | accepts --kv-layout? | accepts FP16 baseline? |
| ------ | -------------- | -------------------- | ---------------------- |
| `mlx_lm.generate` (Python) | mlx-lm 0.31.2 | no `--kv-layout`; native MLX `--kv-bits/--kv-group-size/--quantized-kv-start` only | yes (default; FP16/BF16 K and V) |
| `vllm-swift serve` | 0.3.0 (vLLM 0.19.1) | no TQ+/turbo flag; upstream `--kv-cache-dtype {auto,bf16,fp16,fp8,fp8_e4m3,fp8_e5m2,fp8_inc,fp8_ds_mla}` only | yes via `--dtype float16` |

No `vllm-swift generate` subcommand exists (serve-only), and `mlx-swift-lm` is not installed locally.

## Cached models found

Small enough to load (≤4B), already on disk under `~/.cache/huggingface/hub/`:

- `mlx-community/Qwen3.5-2B-4bit` (1.6 GB) — used
- `mlx-community/Qwen3.5-4B-4bit` (2.9 GB)
- `mlx-community/Qwen2.5-3B-Instruct-4bit` (1.7 GB)
- `mlx-community/Qwen3.5-0.8B-4bit` (0.6 GB)
- `Qwen/Qwen3-0.6B` (1.5 GB, BF16)

Empty refs only (would re-download): `Qwen3-0.6B-4bit`, `Qwen3-4B-4bit`. Skipped.

## Measurement: mlx-community/Qwen3.5-2B-4bit @ 8012 tokens

Engine: `mlx_lm.generate`, `--temp 0.0 --max-tokens 3`, prompt fed via stdin (~7.2K words → 8012 tokens).

| layout | KV cache size | decode tps | prefill tps | notes |
| ------ | ------------- | ---------- | ----------- | ----- |
| FP16 K + FP16 V (default) | 8192 KB / layer at 8K ctx | 453.6 tok/s | 10669.9 tok/s | peak mem 2.678 GB |
| MLX kv-bits=8, group=64 | same alloc; quantized chunk 4 KB | 427.0 tok/s | 10803.2 tok/s | peak mem 2.631 GB |

Short-context sanity (18-token prompt, 3-token decode): FP16 506 tok/s decode, kv-bits=8 492 tok/s decode, peak 1.15 GB both.

## Engine output samples (truncated)

```log
# mlx_lm.generate — FP16 baseline @ 8K
[CC] #1 8192KB
[CC] #100 8192KB
[EVAL-PROFILE] gpu_ops=971 dispatch=154.7ms finalize=0.0ms
Prompt: 8012 tokens, 10669.901 tokens-per-sec
Generation: 3 tokens, 453.598 tokens-per-sec
Peak memory: 2.678 GB

# mlx_lm.generate — kv-bits=8 @ 8K
[CC] #1 8192KB
[CC] #200 4KB
Prompt: 8012 tokens, 10803.196 tokens-per-sec
Generation: 3 tokens, 427.016 tokens-per-sec
Peak memory: 2.631 GB

# vllm-swift version
vllm-swift 0.3.0
dylib: /opt/homebrew/Cellar/vllm-swift/0.3.0/lib/libVLLMBridge.dylib
vLLM: 0.19.1
```

## Verdict

Clean FP16 baseline captured on M5 Max via `mlx_lm.generate` at 8K context (453.6 decode tok/s, 10.7K prefill tok/s, 8192 KB/layer KV, 2.68 GB peak). TQ+ KV layout flags are not wired on either local engine: `mlx_lm` exposes only its native `--kv-bits` MLX integer-quant path, and `vllm-swift` only exposes upstream vLLM `--kv-cache-dtype` (fp8/fp16). Blocker for Stage 2 X bomb mlx-swift / vllm-swift rows: TQ+ runtime ports must land a `--kv-layout` (or equivalent) wiring before we can fill TQ+ asym/centroid-V cells — until then, only FP16 and MLX-native int-quant rows are honestly measurable on this Mac.

<!-- TODO: rerun once TheTom/mlx-swift-lm exposes a tq+ layout flag and once vllm-swift gains a TQ+ kv plugin path. -->
