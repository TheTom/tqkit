"""Backend autodetection.

Looks for known TurboQuant+ inference engines on the user's PATH and
returns version + install info. Used by `tq backends` and as input to
`tq bench` (which dispatches to whichever engines are present).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class BackendInfo:
    name: str
    available: bool
    binary: str | None
    version: str | None
    notes: str = ""


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
        return (out.stdout + out.stderr).strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_llama_cpp() -> BackendInfo:
    binary = shutil.which("llama-cli") or shutil.which("llama-server")
    if binary is None:
        return BackendInfo("llama.cpp", False, None, None,
                           "install from TheTom/llama.cpp@feature/turboquant-kv-cache")
    out = _run([binary, "--version"])
    version = out.splitlines()[0] if out else "unknown"
    return BackendInfo("llama.cpp", True, binary, version)


def detect_vllm() -> BackendInfo:
    try:
        import vllm  # type: ignore
        return BackendInfo("vllm", True, None, getattr(vllm, "__version__", "?"))
    except ImportError:
        return BackendInfo("vllm", False, None, None,
                           "pip install vllm (or use TheTom/vllm forks for TQ+)")


def detect_mlx() -> BackendInfo:
    try:
        import mlx  # type: ignore
        return BackendInfo("mlx", True, None, getattr(mlx, "__version__", "?"))
    except ImportError:
        return BackendInfo("mlx", False, None, None,
                           "pip install mlx (or build TheTom/mlx@feature/turboquant-plus)")


def detect_vllm_swift() -> BackendInfo:
    binary = shutil.which("vllm-swift")
    if binary is None:
        return BackendInfo("vllm-swift", False, None, None,
                           "brew install thetom/vllm-swift")
    out = _run([binary, "--version"])
    version = out.splitlines()[0] if out else "unknown"
    return BackendInfo("vllm-swift", True, binary, version)


def detect_all() -> list[BackendInfo]:
    return [
        detect_llama_cpp(),
        detect_vllm(),
        detect_mlx(),
        detect_vllm_swift(),
    ]
