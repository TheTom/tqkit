"""MLX-Swift engine bridge.

Drives `mlx-swift-lm` (TheTom/mlx-swift-lm@alpha — production-stable,
also drives vllm-swift) to generate against a model with a TurboQuant+
KV cache layout selected.

For tqkit v0.3.x this uses the mlx-swift-lm CLI; future versions may
switch to driving via vllm-swift's OpenAI-compatible endpoint for
parity with how the model is actually served in production.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time

from tqkit.engines.base import Engine, RunResult, EngineNotInstalled


_RE_DECODE_TPS = re.compile(r"(\d+\.?\d*)\s*tokens?-per-sec", re.IGNORECASE)
_RE_KV_MB = re.compile(r"kv\s*cache.*?(\d+\.?\d*)\s*(MB|MiB|GB|GiB)", re.IGNORECASE)


def _bytes_from(num: float, unit: str) -> int:
    mult = {
        "MB":  1000 ** 2, "MiB": 1024 ** 2,
        "GB":  1000 ** 3, "GiB": 1024 ** 3,
    }[unit]
    return int(num * mult)


class MlxSwiftEngine(Engine):
    name = "mlx-swift"

    LAYOUT_FLAGS = {
        "fp16":    ["--kv-layout", "fp16"],
        "tq+sym":  ["--kv-layout", "tq+sym"],
        "tq+asym": ["--kv-layout", "tq+asym"],
    }

    def __init__(self, binary: str | None = None):
        self._binary = binary or shutil.which("mlx-swift-lm")

    def is_installed(self) -> bool:
        return self._binary is not None

    def run(
        self,
        model: str,
        prompt: str,
        ctx_tokens: int,
        layout: str = "fp16",
        max_tokens: int = 64,
        timeout: int = 600,
    ) -> RunResult:
        if not self.is_installed():
            raise EngineNotInstalled(
                "mlx-swift-lm not on PATH. Install from "
                "TheTom/mlx-swift-lm@alpha (the production-stable branch)."
            )
        if not self.supports(layout):
            raise ValueError(f"layout '{layout}' not supported by mlx-swift")

        cmd = [
            self._binary,
            "generate",
            "--model", model,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--max-context", str(ctx_tokens),
            *self.LAYOUT_FLAGS[layout],
        ]

        t0 = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        elapsed = time.time() - t0
        log = (proc.stderr or "") + "\n" + (proc.stdout or "")
        result = RunResult(
            backend=self.name,
            model=model,
            layout=layout,
            ctx_tokens=ctx_tokens,
            output=(proc.stdout or "")[:512],
            total_seconds=elapsed,
            raw_log=log,
        )
        if m := _RE_DECODE_TPS.search(log):
            result.decode_tps = float(m.group(1))
        if m := _RE_KV_MB.search(log):
            result.kv_cache_bytes = _bytes_from(float(m.group(1)), m.group(2))
        return result
