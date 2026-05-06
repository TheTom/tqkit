"""Engine bridges: drive each TurboQuant+ inference backend from one place.

Every engine subclass takes (model, prompt, ctx, layout) and returns a
normalized RunResult. The dispatcher chooses the right subclass based on
the autodetected `BackendInfo.name`.

Output parsing is intentionally lenient: each engine emits its KV-cache
size and decode rate in a slightly different format, but tqkit normalizes
all of them to the same RunResult shape.
"""
from __future__ import annotations

from tqkit.engines.base import Engine, RunResult, EngineNotInstalled
from tqkit.engines.llama_cpp import LlamaCppEngine
from tqkit.engines.vllm_cuda import VllmCudaEngine
from tqkit.engines.vllm_amd import VllmAmdEngine
from tqkit.engines.mlx_swift import MlxSwiftEngine
from tqkit.engines.vllm_swift import VllmSwiftEngine
from tqkit.engines.longctx import LongctxEngine

ENGINES: dict[str, type[Engine]] = {
    "llama-cpp":   LlamaCppEngine,
    "vllm-cuda":   VllmCudaEngine,
    "vllm-amd":    VllmAmdEngine,
    "mlx-swift":   MlxSwiftEngine,
    "vllm-swift":  VllmSwiftEngine,
    "longctx":     LongctxEngine,
}


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise ValueError(
            f"unknown engine '{name}'. known: {sorted(ENGINES)}"
        )
    return ENGINES[name]()


__all__ = [
    "Engine",
    "RunResult",
    "EngineNotInstalled",
    "ENGINES",
    "get_engine",
    "LlamaCppEngine",
    "VllmCudaEngine",
    "VllmAmdEngine",
    "MlxSwiftEngine",
    "VllmSwiftEngine",
    "LongctxEngine",
]
