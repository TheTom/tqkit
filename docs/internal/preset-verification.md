# TurboQuant+ preset bit-width verification (TheTom/vllm)

Source: `/Users/tom/dev/vllm` @ `b572acf9aebc0cc197ef5da9e8b19485e0c97efb` (branch
`feature/turboquant-kv-cache`).

Key files:

- `vllm/config/cache.py:18-43` — `CacheDType` Literal lists all 12 TQ presets.
- `vllm/config/cache.py:245-263` — `_validate_cache_dtype` (purely informational).
- `vllm/utils/torch_utils.py:46-57` — all TQ presets stored as packed `uint8`.
- `vllm/model_executor/layers/quantization/turboquant/config.py:24-66` —
  `_build_presets()` enumerates 6 base configs, each with an `_rv` variant (12 total).
- `config.py:177-220` — `key_packed_size` / `value_packed_size` / `slot_size`.
- `config.py:232-246` — `get_boundary_skip_layers(num_layers, n=2)`.
- `vllm/engine/arg_utils.py:1645-1668` — boundary skip is wired in unconditionally
  for any `cache_dtype.startswith("turboquant_")`, default `n=2`.

All `_bpe` numbers below are **inclusive of all metadata** (per-vector fp16 norm
on keys, per-vector fp16 scale+zero on values). They are derived for
`head_dim=128` (no WHT padding). For `head_dim=256` (Qwen3.6) numbers shift very
slightly because the fixed metadata is amortised over more elements.

## Preset table (head_dim=128)

K key formula: FP8 → `head_dim` B; MSE → `ceil(d*k_bits/8) + 2` B.
V value formula: `ceil(d*v_bits/8) + 4` B. `d = padded_head_dim` if `_rv` else `head_dim`.

| preset                   | K bits | V bits | k_bpe   | v_bpe   | rotation             | skip layers | notes                |
| ------------------------ | ------ | ------ | ------- | ------- | -------------------- | ----------- | -------------------- |
| `turboquant_k8v4`        | 8 (FP8)| 4      | 1.0000  | 0.5313  | none                 | first/last 2 | norm_correction=False |
| `turboquant_k8v3`        | 8 (FP8)| 3      | 1.0000  | 0.4063  | none                 | first/last 2 | norm_correction=False |
| `turboquant_4bit_nc`     | 4 MSE  | 4      | 0.5156  | 0.5313  | WHT on K             | first/last 2 | NC on K               |
| `turboquant_k4v3_nc`     | 4 MSE  | 3      | 0.5156  | 0.4063  | WHT on K             | first/last 2 | NC on K               |
| `turboquant_k3v4_nc`     | 3 MSE  | 4      | 0.3906  | 0.5313  | WHT on K             | first/last 2 | NC on K               |
| `turboquant_3bit_nc`     | 3 MSE  | 3      | 0.3906  | 0.4063  | WHT on K             | first/last 2 | NC on K               |
| `turboquant_k8v4_rv`     | 8 (FP8)| 4      | 1.0000  | 0.5313  | WHT on V (TQ+)       | first/last 2 | k_bpe via FP8 path    |
| `turboquant_k8v3_rv`     | 8 (FP8)| 3      | 1.0000  | 0.4063  | WHT on V (TQ+)       | first/last 2 |                       |
| `turboquant_4bit_nc_rv`  | 4 MSE  | 4      | 0.5156  | 0.5313  | WHT on K + V (TQ+)   | first/last 2 | NC on K               |
| `turboquant_k4v3_nc_rv`  | 4 MSE  | 3      | 0.5156  | 0.4063  | WHT on K + V (TQ+)   | first/last 2 | NC on K               |
| `turboquant_k3v4_nc_rv`  | 3 MSE  | 4      | 0.3906  | 0.5313  | WHT on K + V (TQ+)   | first/last 2 | NC on K               |
| `turboquant_3bit_nc_rv`  | 3 MSE  | 3      | 0.3906  | 0.4063  | WHT on K + V (TQ+)   | first/last 2 | NC on K               |

Note: `_rv` only changes `padded_head_dim` behaviour (matters for non-power-of-2
heads). For `head_dim ∈ {128, 256}` `_rv` does not change `v_bpe` since no
padding is required. `_rv` adds one GEMM/layer at decode but no extra cache
bytes.

## Comparison vs `tqkit/kv_math.py` LAYOUT_BYTES_PER_ELEM

| layout (tqkit) | tqkit (k,v) | nearest vllm preset             | derived (k,v) | drift               | recommendation                       |
| -------------- | ----------- | ------------------------------- | ------------- | ------------------- | ------------------------------------ |
| `fp16`         | (2.0, 2.0)  | n/a (baseline)                  | (2.0, 2.0)    | none                | keep                                 |
| `fp8`          | (1.0, 1.0)  | n/a (vanilla fp8 KV)            | (1.0, 1.0)    | none                | keep                                 |
| `q8_0`         | (1.0, 1.0)  | n/a (llama.cpp legacy)          | ~(1.063, 1.063)| metadata ignored    | keep (legacy, not TQ+)               |
| `tq+sym`       | (1.0, 1.0)  | `turboquant_k8v4` w/ FP8 K + 4bV| (1.0, 0.5313) | **V over-counted**  | **rename / split**: `tq+sym` is misleading; the headline TQ stack is asymmetric |
| `tq+asym`      | (1.0, 0.5)  | `turboquant_k8v4` (or `_rv`)    | (1.0, 0.5313) | v_bpe under-counts ~6.25% | tighten v_bpe → 0.5313             |
| `turbo4`       | (0.5, 0.5)  | `turboquant_4bit_nc(_rv)`       | (0.5156, 0.5313)| both under by ~3–6% | tighten to (0.5156, 0.5313)        |

Drift summary: tqkit currently rounds metadata away. The fp16 norm on K (MSE
modes) and the fp16 scale+zero on V add a fixed 2 B / 4 B per vector, which is
3–6% overhead at `head_dim=128` and ~1.5–3% at `head_dim=256`.

## Boundary layer skip behaviour

`TurboQuantConfig.get_boundary_skip_layers(num_layers, n=2)` returns the first
`n` and last `n` layer indices, with `n` capped at `num_layers // 2`. The call
site in `arg_utils.py:1660` uses the default `n=2` — **always**. There is no
model-dependent override in the current branch. Hybrid (Mamba+attention) models
are rejected entirely (`arg_utils.py:1649-1654`).

So in practice every TQ run with this fork loses the cheap-cache rate on
exactly **4 layers**: layers `{0, 1, num_layers-2, num_layers-1}`. For pinned
models that's:

| model                  | total attn layers | TQ layers | skipped layers |
| ---------------------- | ----------------- | --------- | -------------- |
| qwen2.5-7b             | 28                | 24        | 4              |
| qwen2.5-14b-1m         | 48                | 44        | 4              |
| qwen2.5-32b            | 64                | 60        | 4              |
| qwen3-8b               | 36                | 32        | 4              |
| llama-3.1-8b           | 32                | 28        | 4              |
| llama-3.1-70b          | 80                | 76        | 4              |
| qwen3.6-27b (hybrid)   | 16                | rejected  | n/a            |

## Recommendation for `tqkit/kv_math.py`

1. **Tighten `tq+asym` from `(1.0, 0.5)` → `(1.0, 0.5313)`**. Matches the
   `turboquant_k8v4` headline preset at `head_dim=128`. Rename internal
   comment to clarify it's `K=FP8 + V=4bit + scale+zero`.
2. **Tighten `turbo4` from `(0.5, 0.5)` → `(0.5156, 0.5313)`** (matches
   `turboquant_4bit_nc`). Add a comment that 3-bit variants are
   `(0.3906, 0.4063)`.
3. **Remove or rename `tq+sym`**. There is no symmetric q8_0+WHT preset in the
   vllm fork; the current entry is fictional. Either drop it or alias it to
   `q8_0` (1.0, 1.0).
4. **Boundary skip — recommend option (b)**: document the 4-layer skip in the
   methodology page caveat. Baking it into a `tq+asym (effective)` layout would
   couple `kv_math.py` to model topology (and the skip is `min(n, L//2)`, so it
   doesn't generalise as a single multiplier). One footnote per benchmark table
   is cleaner: "TQ entries skip first/last 2 attention layers in the vLLM
   fork; the reported per-token bytes assume all layers are quantised, so true
   savings are 1–8% lower than tabulated."
5. Optionally add `tq+asym_3bit` entries (`(1.0, 0.4063)`, `(0.3906, 0.5313)`,
   etc.) if benchmarking the 3-bit family becomes a thing.
