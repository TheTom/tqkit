"""KV cache size math.

Per-token KV cache bytes for a transformer with GQA:
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element

The 2 is for K and V. For TurboQuant+ asymmetric layouts, K and V can have
different per-element costs (e.g. K=q8_0=1 B, V=turbo4=0.5 B effective).

Reference values pulled from each model's HuggingFace config.json. Pinned
here to avoid runtime HF API calls so `tq report` works offline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArch:
    """Subset of HF config needed for KV cache math.

    For hybrid linear+full attention models (Qwen3-Next, Qwen3.6), only the
    full-attention layers contribute to the standard KV cache; linear-attention
    layers carry their own recurrent state. So `num_layers` here means
    "number of layers that materialize a standard KV cache," not the total
    transformer depth. The `notes` field flags hybrid architectures.
    """
    name: str
    num_layers: int
    num_kv_heads: int  # GQA: distinct from attention heads
    head_dim: int
    notes: str = ""


# Pinned model architectures used in TurboQuant+ benchmarking.
MODELS: dict[str, ModelArch] = {
    "qwen2.5-7b-instruct": ModelArch(
        "Qwen/Qwen2.5-7B-Instruct",
        num_layers=28, num_kv_heads=4, head_dim=128,
    ),
    "qwen2.5-14b-instruct-1m": ModelArch(
        "Qwen/Qwen2.5-14B-Instruct-1M",
        num_layers=48, num_kv_heads=8, head_dim=128,
    ),
    "qwen2.5-32b-instruct": ModelArch(
        "Qwen/Qwen2.5-32B-Instruct",
        num_layers=64, num_kv_heads=8, head_dim=128,
    ),
    "qwen3-8b": ModelArch(
        "Qwen/Qwen3-8B",
        num_layers=36, num_kv_heads=8, head_dim=128,
    ),
    "qwen3-next-80b-a3b": ModelArch(
        "Qwen/Qwen3-Next-80B-A3B",
        num_layers=48, num_kv_heads=2, head_dim=128,  # MoE attention layers
        notes="MoE; only the attention layers contribute to KV cache",
    ),
    "qwen3.6-27b": ModelArch(
        "Qwen/Qwen3.6-27B",
        # 64 layers total: 48 linear_attention + 16 full_attention.
        # Only the 16 full_attention layers carry a standard KV cache.
        num_layers=16, num_kv_heads=4, head_dim=256,
        notes="hybrid linear+full attention; KV cache from 16/64 layers only",
    ),
    "qwen3.6-35b-a3b": ModelArch(
        "Qwen/Qwen3.6-35B-A3B",
        # 40 layers, hybrid; the full-attention count from layer_types is
        # used here. MoE: 256 experts, 8 active per token (~3B active).
        num_layers=10, num_kv_heads=2, head_dim=256,
        notes="MoE + hybrid linear/full attention; ~10 attention layers",
    ),
    "llama-3.1-8b-instruct": ModelArch(
        "meta-llama/Llama-3.1-8B-Instruct",
        num_layers=32, num_kv_heads=8, head_dim=128,
    ),
    "llama-3.1-70b-instruct": ModelArch(
        "meta-llama/Llama-3.1-70B-Instruct",
        num_layers=80, num_kv_heads=8, head_dim=128,
    ),
    "mistral-7b-instruct-v0.3": ModelArch(
        "mistralai/Mistral-7B-Instruct-v0.3",
        num_layers=32, num_kv_heads=8, head_dim=128,
    ),
}


# Bytes per element per layout (per K side and per V side, summed elsewhere).
LAYOUT_BYTES_PER_ELEM: dict[str, tuple[float, float]] = {
    "fp16":         (2.0, 2.0),    # K, V both FP16
    "fp8":          (1.0, 1.0),    # K=fp8, V=fp8
    "q8_0":         (1.0, 1.0),    # K=q8_0, V=q8_0 (legacy llama.cpp)
    "tq+sym":       (1.0, 1.0),    # K=q8_0+WHT, V=q8_0+WHT
    "tq+asym":      (1.0, 0.5),    # K=q8_0+WHT, V=turbo4 (the headline layout)
    "turbo4":       (0.5, 0.5),    # K=turbo4, V=turbo4 (very aggressive)
}


def kv_bytes_per_token(arch: ModelArch, layout: str) -> float:
    """Per-token KV cache bytes for a given layout."""
    if layout not in LAYOUT_BYTES_PER_ELEM:
        raise ValueError(
            f"unknown layout '{layout}'. "
            f"known: {sorted(LAYOUT_BYTES_PER_ELEM)}"
        )
    k_bpe, v_bpe = LAYOUT_BYTES_PER_ELEM[layout]
    elems = arch.num_layers * arch.num_kv_heads * arch.head_dim
    return elems * (k_bpe + v_bpe)


def kv_bytes_total(arch: ModelArch, layout: str, ctx: int) -> float:
    """Total KV cache bytes at a given context length."""
    return kv_bytes_per_token(arch, layout) * ctx


def fmt_bytes(b: float) -> str:
    """Human-readable byte size with 1-decimal precision."""
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]
    for unit, scale in units:
        if b >= scale:
            return f"{b / scale:.1f} {unit}"
    return f"{b:.0f} B"


def savings_pct(baseline_bytes: float, target_bytes: float) -> float:
    """Percent reduction from baseline to target."""
    if baseline_bytes <= 0:
        return 0.0
    return (1.0 - target_bytes / baseline_bytes) * 100.0
