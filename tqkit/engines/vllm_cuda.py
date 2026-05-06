"""vLLM CUDA engine bridge.

Calls `vllm bench` (or `vllm serve` + a probe) with --kv-cache-dtype set
to the chosen layout. Parses standard vllm log lines for KV cache size +
decode tps.

For tqkit v0.3.x this uses the `vllm bench latency` subcommand which
emits a JSON-like summary; future versions may switch to direct Python
import for tighter integration.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time

from tqkit.engines.base import Engine, RunResult, EngineNotInstalled


_RE_KV_GIB = re.compile(r"# of GPU blocks:\s*\d+.*?KV cache.*?(\d+\.?\d*)\s*GiB", re.IGNORECASE | re.DOTALL)
_RE_KV_GB_LINE = re.compile(r"KV cache.*?(\d+\.?\d*)\s*GiB")
_RE_DECODE_TPS = re.compile(r"output_tokens?_per_second[\":\s=]+(\d+\.?\d*)", re.IGNORECASE)
_RE_DECODE_TPS_ALT = re.compile(r"Avg generation throughput:\s*(\d+\.?\d*)\s*tokens?/s", re.IGNORECASE)


class VllmCudaEngine(Engine):
    name = "vllm-cuda"

    # vLLM standard --kv-cache-dtype values + the patched TQ+ values from
    # TheTom/vllm@feature/turboquant-kv-cache.
    # Maps tqkit canonical layout names to the actual --kv-cache-dtype
    # values accepted by TheTom/vllm@feature/turboquant-kv-cache. The TQ+
    # presets encode bit widths into the name itself (k8v4 = K=8bit V=4bit;
    # 4bit_nc = K=4 V=4 sym, no centroid).
    LAYOUT_FLAGS = {
        "fp16":    ["--kv-cache-dtype", "auto"],
        "fp8":     ["--kv-cache-dtype", "fp8"],
        "tq+sym":  ["--kv-cache-dtype", "turboquant_4bit_nc"],
        "tq+asym": ["--kv-cache-dtype", "turboquant_k8v4"],
    }

    def __init__(self, binary: str | None = None):
        self._binary = binary or shutil.which("vllm")

    def is_installed(self) -> bool:
        if self._binary is not None:
            return True
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

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
                "vllm not installed. pip install vllm or use "
                "TheTom/vllm@feature/turboquant-kv-cache."
            )
        if not self.supports(layout):
            raise ValueError(f"layout '{layout}' not supported by vllm-cuda")

        cmd = [
            self._binary or "vllm",
            "bench", "latency",
            "--model", model,
            "--input-len", str(min(ctx_tokens, 4096)),  # bench harness, not full ctx
            "--output-len", str(max_tokens),
            "--num-iters", "1",
            "--max-model-len", str(ctx_tokens),
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
        for rx in (_RE_KV_GIB, _RE_KV_GB_LINE):
            if m := rx.search(log):
                result.kv_cache_bytes = int(float(m.group(1)) * (1024 ** 3))
                break
        for rx in (_RE_DECODE_TPS, _RE_DECODE_TPS_ALT):
            if m := rx.search(log):
                result.decode_tps = float(m.group(1))
                break
        return result
