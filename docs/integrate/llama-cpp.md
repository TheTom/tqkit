# Integrating TurboQuant+ into llama.cpp

Use TurboQuant+ KV-cache compression with [llama.cpp](https://github.com/ggml-org/llama.cpp) on CUDA, Metal, ROCm, or CPU.

## Install

Clone the patched fork and build:

```bash
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON   # macOS Apple Silicon
# or  -DGGML_CUDA=ON              # NVIDIA CUDA
# or  -DGGML_HIPBLAS=ON            # AMD ROCm
cmake --build build -j
```

## Run with TurboQuant+ KV cache

The TQ+ layouts are exposed as `--ctk` (key) and `--ctv` (value) flags:

```bash
./build/bin/llama-cli \
    -m models/Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf \
    -c 32768 \
    --ctk q8_0 \
    --ctv turbo4 \
    -f my_prompt.txt \
    --n-predict 256
```

### Layout flags

| layout name | `--ctk` | `--ctv` | per-token KV (Qwen2.5-14B) | savings vs FP16 |
| ----------- | ------- | ------- | -------------------------- | --------------- |
| `fp16` baseline | `f16` | `f16` | 192 KB | — |
| `tq+sym` | `q8_0` | `q8_0` | 96 KB | 50% |
| `tq+asym` (headline) | `q8_0` | `turbo4` | 72 KB | 62.5% |
| `tq+aggressive` | `q4_0` | `turbo4` | 48 KB | 75% |

`tq+asym` is the recommended default — minimal PPL drift (+0.05–0.20 absolute) at 62.5% savings.

## Verify the savings

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 32K --layout tq+asym
# → 2.2 GB total vs 6.0 GB FP16 (62.5% savings)
```

## Caveats

- **Model compatibility**: TurboQuant requires power-of-2 `head_dim`. Qwen2.5 (head_dim=128) ✓. Qwen3-4B (head_dim=80) ✗ — patch is on the roadmap.
- **Qwen2 models** need Q/K/V attention bias loaded; otherwise WHT silently fails. Qwen2.5 onwards is fine.
- **Hybrid models** (e.g. Qwen3-Next): use `tq+sym` only, not `tq+asym`. The MoE path is sensitive to V-side aggressive quantization.

## Verifying correctness

Pick a known prompt with deterministic output (e.g. low-temperature factual Q&A). Compare output between FP16 and TQ+ runs. PPL on wikitext-2-raw should drift ≤0.20 absolute at the same context length.

For full perplexity validation:

```bash
./build/bin/llama-perplexity \
    -m models/Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --ctk q8_0 --ctv turbo4 \
    -c 8192
```
