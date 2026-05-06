# How `tqkit` verifies KV-cache compression claims

`tqkit` is built so that anyone — adopter, skeptic, eval team, or competitor — can independently verify a KV-cache compression claim *before* trusting any benchmark headline. This page is the methodology.

## Two layers of verification

The toolkit measures KV-cache savings at **two layers**, and each catches a different class of overclaim.

| layer | tool | catches |
| ----- | ---- | ------- |
| 1. theoretical math | `tq report`, `tq table` | impossible-by-arithmetic claims (e.g. "75% savings on FP16 KV" while only quantizing V from 16 to 8 bit) |
| 2. runtime measurement | `tq bench` (engine bridges) | implementation overhead, hidden metadata, boundary-layer skips, real-vs-claimed PPL drift |

Skeptics can run layer 1 in 5 seconds without installing any inference engine; layer 2 needs the matching hardware + a real model.

## Layer 1 — theoretical math

The KV cache footprint of any standard transformer is fully determined by its architecture:

```
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

The factor of 2 is for K and V. For asymmetric layouts where K and V have different per-element costs (e.g. K=8 bit, V=4 bit), substitute `(k_bpe + v_bpe)` for `2 * bytes_per_element`.

`tqkit` ships pinned `ModelArch` values for every model it supports. Run `tq report --model X --layout L` to get the exact bytes/token and total at any context length. The output does not depend on whether the engine is even installed.

### What this catches

A claim of "75% KV cache savings" implies the per-element cost dropped from 16 bit to ~4 bit. That's only possible if both K and V are quantized to roughly 4 bit each. Any compression scheme where K stays at 8 bit and V drops to 4 bit (the canonical "asymmetric" layout) gives **62.5%** savings — not 75%.

Specifically, on a Qwen2.5-14B model:

| scheme | K bpe | V bpe | bytes/token | savings |
| ------ | ----- | ----- | ----------- | ------- |
| FP16 | 2 | 2 | 192 KB | — |
| K=q8_0, V=fp16 (legacy) | 1 | 2 | 144 KB | 25% |
| K=q8_0, V=q8_0 (sym 8-bit) | 1 | 1 | 96 KB | 50% |
| **K=q8_0, V=turbo4 (TQ+ asym)** | **1** | **0.5** | **72 KB** | **62.5%** |
| K=q4_0, V=q4_0 (sym 4-bit) | 0.5 | 0.5 | 48 KB | 75% |
| K=q3, V=q3 (sym 3-bit, aggressive) | 0.375 | 0.375 | 36 KB | 81% |

If a paper reports "70% savings" without specifying their (K_bpe, V_bpe) pair, the math will tell you which row they're closest to. Reproduce with: `tq table --model qwen2.5-14b-instruct-1m`.

### Hybrid models — what to count

For hybrid linear+full attention models (Qwen3-Next, Qwen3.6) only the **full-attention layers** carry a standard KV cache. `ModelArch.num_layers` in tqkit refers to attention-layer count, not total transformer depth. The `notes` field flags every hybrid model.

Running `tq report --model qwen3.6-27b --ctx 256K --layout fp16` returns 16 GB total — which corresponds to 16/64 layers, not 64/64. A claim that doesn't account for the linear-attention layers will overstate KV cache size by ~4× on this architecture.

**vLLM TurboQuant+ rejects hybrid models entirely** (per `vllm/engine/arg_utils.py` in TheTom/vllm@feature/turboquant-kv-cache). If a competitor claims TQ-style savings on Qwen3-Next, they're either using a different implementation or the claim doesn't apply to the standard fork.

### Boundary-layer skip

The vllm TurboQuant+ fork keeps the **first and last 2 attention layers at FP16 unconditionally**, regardless of model. This is `get_boundary_skip_layers(num_layers, n=2)` in `vllm/model_executor/layers/quantization/turboquant/config.py`.

For a 48-layer model, that's 4 FP16 layers + 44 quantized layers. Real-world savings are 5–8% lower than the all-quantized math `tq report` reports. To compute the realistic figure:

```python
from tqkit.kv_math import kv_bytes_total_with_boundary_skip, MODELS
arch = MODELS["qwen2.5-14b-instruct-1m"]
realistic_bytes = kv_bytes_total_with_boundary_skip(
    arch, layout="tq+asym", ctx=1_000_000, n_skip=2,
)
```

For the canonical Qwen2.5-14B at 1M context with `tq+asym`, the all-quantized headline is **72 GB** (61.7% savings) but the boundary-skip-adjusted reality is **~83 GB** (57% savings). Both numbers fit on a single MI300X (192 GB after weights), but the post should report the realistic 57%, not the headline 61.7%, when claiming "fits on one GPU." Otherwise the audit step 1 (math vs claim) flags it.

### Per-element metadata overhead

A "4-bit" V is not literally 0.5 bytes per element — every block carries an FP16 scale + zero point. At `head_dim=128`, real V_bpe is 0.5313 (5.9% overhead). At `head_dim=256` it's closer to 0.516. Verified bpe values for every TurboQuant+ preset live in `tqkit/kv_math.py:LAYOUT_BYTES_PER_ELEM` with cross-reference to vllm source.

## Layer 2 — runtime measurement

Theoretical math says what the cache *should* be. `tq bench` runs the actual engine and verifies what the cache *is*. Two things the math cannot catch:

1. **Hidden metadata overhead.** Every quantization scheme stores scales, zero points, codebooks, etc. A "4-bit" quantization with per-block scales actually averages 0.5–0.55 bytes per element. tqkit reports the engine's own measured size, not the math-only ideal.
2. **Boundary-layer skips.** Many TQ+ implementations skip the first/last 2 layers (where quantization hurts most). If a paper claims "65% savings" but skips 4 of 48 layers, the *real* per-token cost is 4/48 layers × 2 bytes (FP16) + 44/48 × 0.6 bytes (TQ+) = much higher than naive math suggests.

`tq bench` parses each engine's KV cache size from its own log output:

| engine | parser source |
| ------ | ------------- |
| llama.cpp | `KV self size = N MiB` line in stderr |
| vLLM (CUDA + AMD) | `KV cache size: N GiB` initialization line |
| MLX-Swift | `kv cache: N GiB allocated` |
| vllm-swift | `[vllm-swift] kv-cache: N GB` |

If the measured number drifts from the theoretical number by more than ~5%, the engine is either skipping layers, using a different layout than advertised, or has metadata overhead the paper didn't disclose.

## Audit checklist for evaluating a KV-compression paper

When a new paper claims X% KV cache savings:

1. Read off their `(K_bpe, V_bpe)` pair (or compute it from their bit-width claim). Plug into the per-token formula. Does the math match their headline? If not, **someone is rounding generously or skipping layers without disclosure.**
2. Check `head_dim`. Does their scheme need power-of-2 head_dim? If yes, models like Qwen3-4B (head_dim=80) are silently incompatible.
3. For hybrid models, ask "which layers carry KV cache?" If the paper benchmarks on a hybrid model and quotes "savings vs FP16 baseline," verify they're comparing apples to apples — both layouts should count the same layer subset.
4. Run `tq bench` on their target engine + model + layout. Does the measured KV cache match their reported size within 5%?
5. Compare PPL drift on a known eval set (wikitext-2-raw is the standard). Is it within their claimed range?

A paper that survives this audit is honest. A paper that flunks step 1 isn't going to survive 4 or 5 either.

## Why this matters

Long-context inference is now a marketing battlefield. Multiple closed-stack startups raise rounds claiming "subquadratic" or "10x KV savings" architectures. Most of those claims are either:

- True but trivial (50% savings = K and V both at 8-bit, which has been standard since 2024)
- Overstated (75% savings claimed; real number is ~62.5%)
- Contingent on cherry-picked workloads (great PPL on wikitext, falls apart on long-context retrieval like MRCR)

`tqkit` exists because the open ecosystem deserves a way to fact-check long-context claims as fast as it can fact-check a chess engine's Elo: take the claim, run the canonical bench, compare against the headline, publish the receipts. The default should be "show me", not "trust me."
