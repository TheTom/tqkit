# Integrating TurboQuant+ into your inference engine

One-page recipes for plugging TurboQuant+ KV-cache compression into each supported backend.

| backend | hardware | doc |
| ------- | -------- | --- |
| llama.cpp | NVIDIA / Apple / AMD / CPU | [llama-cpp.md](llama-cpp.md) |
| vLLM (NVIDIA) | NVIDIA datacenter + consumer | [vllm-cuda.md](vllm-cuda.md) |
| vLLM (AMD ROCm) | MI300X / MI250X | [vllm-amd.md](vllm-amd.md) |
| MLX-Swift | Apple Silicon (M-series, iPhone) | [mlx-swift.md](mlx-swift.md) |
| vllm-swift | Apple Silicon as OpenAI-API server | [vllm-swift.md](vllm-swift.md) |

## Same idea everywhere

Each engine exposes a flag that swaps the KV cache from FP16 to a TurboQuant+ layout:

| engine | flag |
| ------ | ---- |
| llama.cpp | `--ctk q8_0 --ctv turbo4` |
| vLLM (CUDA / AMD) | `--kv-cache-dtype tq_asym` |
| MLX-Swift | `TurboKVCache(.asymmetric)` |
| vllm-swift | `--kv-layout tq+asym` |

The math is the same in every case: for a 14B model with 8 KV heads × 128 head_dim × 48 layers, switching from FP16 (192 KB/token) to TQ+ asym (72 KB/token) saves 62.5% of KV cache memory at any context length.

## Verify before you commit

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 32K --layout tq+asym
tq table --model qwen2.5-14b-instruct-1m
```

The math is purely architectural — anyone can verify the savings claim before installing a single inference engine.
