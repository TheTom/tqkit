# Integrating TurboQuant+ into MLX-Swift

Use TurboQuant+ KV-cache compression with [MLX-Swift](https://github.com/ml-explore/mlx-swift) on Apple Silicon (M-series Macs and iPhones).

## Install

The TurboQuant+ port lives on a fork of [mlx](https://github.com/TheTom/mlx) and a fork of [mlx-swift-lm](https://github.com/TheTom/mlx-swift-lm) with matching cache types.

In your `Package.swift`:

```swift
.package(url: "https://github.com/TheTom/mlx-swift-lm.git", branch: "feature/turboquant-plus")
```

Or for direct mlx (Python):

```bash
pip install --upgrade git+https://github.com/TheTom/mlx@feature/turboquant-plus
```

## Use TurboQuant+ KV cache

```swift
import MLX
import MLXLM

let model = try await loadModel(named: "qwen2.5-14b-instruct-1m-4bit")

// Default cache: KVCacheSimple (FP16).
// TurboQuant+ alternative:
let cache = TurboKVCache(layers: model.numLayers,
                         kvHeads: model.numKVHeads,
                         headDim: model.headDim,
                         layout: .asymmetric)  // K=q8_0+WHT, V=turbo4

let result = try await generate(
    model: model,
    prompt: "What is the capital of France?",
    cache: cache,
    maxTokens: 256
)
```

## Layouts

| layout case | K layout | V layout | savings vs FP16 |
| ----------- | -------- | -------- | --------------- |
| `.fp16` (default) | FP16 | FP16 | — |
| `.symmetric` | q8_0+WHT | q8_0+WHT | 50% |
| `.asymmetric` (headline) | q8_0+WHT | turbo4 | 62.5% |

## Verify the savings

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 32K --layout tq+asym
```

## Validated numbers

From the `feature/turboquant-plus` branch (Apple M5 Max, 2026-04):

| metric | FP16 baseline | TQ+ asym | delta |
| ------ | ------------- | -------- | ----- |
| Decode tps | 67 | 58 | -13% |
| Wikitext-2 PPL @ 8K | 6.62 | 6.71 | +0.09 |
| KV cache @ 32K | 6.0 GB | 2.25 GB | -62.5% |

## On iPhone

The Pal iOS app uses this path. Tested on iPhone 16 Pro (A18 Pro, 8 GB RAM) — TurboQuant+ asymmetric KV is what makes longer-context chat sessions fit in available memory before OOM.

## Caveats

- MLX-Swift is the production target. The Python `mlx` fork is for prototyping.
- WHT requires power-of-2 `head_dim` (Qwen2.5 ✓, Qwen3-4B ✗).
- `KVCacheTurbo` shapes match `KVCacheSimple` for drop-in replacement; multi-batch (BatchedKVCache) supported on `feature/turboquant-plus`.
