"""vLLM AMD ROCm engine bridge.

Same shape as VllmCudaEngine — vLLM uses the same `--kv-cache-dtype` flag
on ROCm as CUDA. The differentiator is the underlying kernel selection
(MFMA-friendly scoring on MI300X) and the docker image. From tqkit's
orchestration layer's perspective the runtime call is identical.
"""
from __future__ import annotations

import shutil

from tqkit.engines.vllm_cuda import VllmCudaEngine


class VllmAmdEngine(VllmCudaEngine):
    name = "vllm-amd"

    def is_installed(self) -> bool:
        # Same Python-level vLLM check; in addition, expect ROCm to be
        # present. We do NOT hard-fail on missing ROCm tooling here —
        # the engine binary itself will surface that error if it can't
        # initialize a device.
        if shutil.which("rocm-smi") is None:
            return False
        return super().is_installed()
