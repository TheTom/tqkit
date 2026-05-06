"""llama.cpp engine bridge.

Calls `llama-cli` with the right --ctk/--ctv flags for each layout. Parses
the standard llama.cpp stderr to extract decode tps and KV cache size.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time

from tqkit.engines.base import Engine, RunResult, EngineNotInstalled


_RE_DECODE_TPS = re.compile(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs?\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
_RE_KV_SIZE = re.compile(r"KV self size\s*=\s*([\d.]+)\s*(MiB|MB|GiB|GB)")
_RE_PROMPT_TPS = re.compile(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*tokens?\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")


def _bytes_from(num: float, unit: str) -> int:
    mult = {
        "MiB": 1024 ** 2,
        "MB":  1000 ** 2,
        "GiB": 1024 ** 3,
        "GB":  1000 ** 3,
    }[unit]
    return int(num * mult)


class LlamaCppEngine(Engine):
    name = "llama-cpp"

    LAYOUT_FLAGS = {
        "fp16":         ["--ctk", "f16",  "--ctv", "f16"],
        "q8_0":         ["--ctk", "q8_0", "--ctv", "q8_0"],
        "tq+sym":       ["--ctk", "q8_0", "--ctv", "q8_0"],
        "tq+asym":      ["--ctk", "q8_0", "--ctv", "turbo4"],
        "turbo4":       ["--ctk", "turbo4", "--ctv", "turbo4"],
    }

    def __init__(self, binary: str | None = None):
        self._binary = binary or (shutil.which("llama-cli")
                                  or shutil.which("llama-server"))

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
                "llama-cli not on PATH. Install from "
                "TheTom/llama.cpp@feature/turboquant-kv-cache."
            )
        if not self.supports(layout):
            raise ValueError(f"layout '{layout}' not supported by llama.cpp")

        cmd = [
            self._binary,
            "-m", model,
            "-c", str(ctx_tokens),
            "-n", str(max_tokens),
            "-p", prompt,
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
        if m := _RE_PROMPT_TPS.search(log):
            result.prefill_tps = float(m.group(1))
        if m := _RE_KV_SIZE.search(log):
            result.kv_cache_bytes = _bytes_from(float(m.group(1)), m.group(2))
        return result
