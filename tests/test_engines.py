"""Tests for engine bridges.

Mocks `subprocess.run` so we don't actually launch llama-cli, vllm, etc.
The point is to verify the dispatcher chooses the right binary, builds
the right command line, and parses representative output correctly.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tqkit.engines import (
    ENGINES,
    EngineNotInstalled,
    LlamaCppEngine,
    LongctxEngine,
    MlxSwiftEngine,
    VllmAmdEngine,
    VllmCudaEngine,
    VllmSwiftEngine,
    get_engine,
)


# --- Dispatcher / smoke ---

def test_get_engine_known():
    assert isinstance(get_engine("llama-cpp"), LlamaCppEngine)
    assert isinstance(get_engine("vllm-cuda"), VllmCudaEngine)
    assert isinstance(get_engine("vllm-amd"), VllmAmdEngine)
    assert isinstance(get_engine("mlx-swift"), MlxSwiftEngine)
    assert isinstance(get_engine("vllm-swift"), VllmSwiftEngine)
    assert isinstance(get_engine("longctx"), LongctxEngine)


def test_get_engine_unknown_raises():
    with pytest.raises(ValueError):
        get_engine("not-a-real-engine")


def test_engines_keys_match_subclass_names():
    for name, cls in ENGINES.items():
        assert cls.name == name


# --- Layout support ---

def test_llama_cpp_supports_canonical_layouts():
    e = LlamaCppEngine(binary="/fake/llama-cli")
    assert e.supports("fp16")
    assert e.supports("tq+asym")
    assert e.supports("tq+sym")
    assert not e.supports("not-a-layout")


def test_vllm_cuda_layout_flags_use_real_preset_names():
    """vLLM TQ+ presets encode bit widths in the name; tq_asym/tq_sym
    aliases don't exist in the actual fork."""
    e = VllmCudaEngine(binary="/fake/vllm")
    assert e.LAYOUT_FLAGS["tq+asym"] == ["--kv-cache-dtype", "turboquant_k8v4"]
    assert e.LAYOUT_FLAGS["tq+sym"]  == ["--kv-cache-dtype", "turboquant_4bit_nc"]


# --- Not-installed paths ---

def test_llama_cpp_run_raises_when_not_installed():
    e = LlamaCppEngine(binary=None)
    assert not e.is_installed()
    with pytest.raises(EngineNotInstalled):
        e.run(model="m", prompt="p", ctx_tokens=1024)


def test_vllm_cuda_run_raises_when_not_installed():
    e = VllmCudaEngine(binary=None)
    with patch.dict("sys.modules", {"vllm": None}), \
         patch("tqkit.engines.vllm_cuda.shutil.which", return_value=None):
        e2 = VllmCudaEngine()
        # Force the importable check to fail too:
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "vllm":
                raise ImportError("no vllm")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fake_import):
            assert not e2.is_installed()


def test_unsupported_layout_raises():
    e = LlamaCppEngine(binary="/fake/llama-cli")
    with pytest.raises(ValueError, match="layout"):
        e.run(model="m", prompt="p", ctx_tokens=1024, layout="bogus")


# --- Output parsing ---

LLAMA_LOG = """
build: 4321 (abc1234) with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0
llm_load_tensors: ggml ctx size = 0.27 MiB
llama_kv_cache_init:        CPU KV buffer size =   256.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (q8_0): 512.00 MiB, V (turbo4): 512.00 MiB
prompt eval time =     220.45 ms /   100 tokens (    2.20 ms per token,   453.65 tokens per second)
       eval time =    1234.56 ms /    64 runs   (   19.29 ms per token,    51.84 tokens per second)
"""


def _fake_proc(stderr=LLAMA_LOG, stdout="", returncode=0):
    return SimpleNamespace(stderr=stderr, stdout=stdout, returncode=returncode)


def test_llama_cpp_parses_decode_tps_and_kv_size():
    e = LlamaCppEngine(binary="/fake/llama-cli")
    with patch("tqkit.engines.llama_cpp.subprocess.run",
               return_value=_fake_proc()):
        result = e.run(model="m", prompt="p", ctx_tokens=2048,
                       layout="tq+asym")
    assert result.backend == "llama-cpp"
    assert result.layout == "tq+asym"
    assert result.decode_tps == pytest.approx(51.84, rel=1e-3)
    assert result.prefill_tps == pytest.approx(453.65, rel=1e-3)
    # 1024 MiB
    assert result.kv_cache_bytes == 1024 * (1024 ** 2)


VLLM_LOG = """
INFO 01-01 ... Initializing engine
INFO 01-01 ... # of GPU blocks: 5120, KV cache size: 6.40 GiB
INFO 01-01 ... Avg generation throughput: 87.5 tokens/s
"""


def test_vllm_cuda_parses_kv_size_and_decode_tps():
    e = VllmCudaEngine(binary="/fake/vllm")
    with patch("tqkit.engines.vllm_cuda.subprocess.run",
               return_value=_fake_proc(stderr=VLLM_LOG)):
        result = e.run(model="m", prompt="p", ctx_tokens=32768,
                       layout="tq+asym")
    assert result.backend == "vllm-cuda"
    assert result.kv_cache_bytes == int(6.40 * (1024 ** 3))
    assert result.decode_tps == pytest.approx(87.5)


def test_vllm_cuda_command_line_uses_correct_kv_dtype():
    """`tq+asym` should map to --kv-cache-dtype turboquant_k8v4."""
    e = VllmCudaEngine(binary="/fake/vllm")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(stderr="")

    with patch("tqkit.engines.vllm_cuda.subprocess.run",
               side_effect=fake_run):
        e.run(model="m", prompt="p", ctx_tokens=1024, layout="tq+asym")
    cmd = captured["cmd"]
    assert "--kv-cache-dtype" in cmd
    assert "turboquant_k8v4" in cmd


# --- vllm-amd inherits from CUDA ---

def test_vllm_amd_inherits_layout_flags():
    e = VllmAmdEngine(binary="/fake/vllm")
    assert e.LAYOUT_FLAGS == VllmCudaEngine.LAYOUT_FLAGS
    assert e.name == "vllm-amd"


def test_vllm_amd_requires_rocm_smi():
    """is_installed() returns False if rocm-smi isn't on PATH."""
    e = VllmAmdEngine(binary="/fake/vllm")
    with patch("tqkit.engines.vllm_amd.shutil.which", return_value=None):
        assert not e.is_installed()


# --- mlx-swift ---

MLX_LOG = """
loading model from qwen2.5-14b-instruct-1m-4bit...
generated 64 tokens at 67.3 tokens-per-sec
kv cache: 2.25 GiB allocated
"""


def test_mlx_swift_parses_decode_tps_and_kv_size():
    e = MlxSwiftEngine(binary="/fake/mlx-swift-lm")
    with patch("tqkit.engines.mlx_swift.subprocess.run",
               return_value=_fake_proc(stderr=MLX_LOG)):
        result = e.run(model="m", prompt="p", ctx_tokens=32768,
                       layout="tq+asym")
    assert result.backend == "mlx-swift"
    assert result.decode_tps == pytest.approx(67.3)
    assert result.kv_cache_bytes == int(2.25 * (1024 ** 3))


def test_mlx_swift_run_raises_when_not_installed():
    e = MlxSwiftEngine(binary=None)
    with pytest.raises(EngineNotInstalled):
        e.run(model="m", prompt="p", ctx_tokens=1024)


def test_mlx_swift_unsupported_layout():
    e = MlxSwiftEngine(binary="/fake/mlx-swift-lm")
    with pytest.raises(ValueError, match="layout"):
        e.run(model="m", prompt="p", ctx_tokens=1024, layout="bogus")


# --- vllm-swift ---

VLLM_SWIFT_LOG = """
[vllm-swift] serving qwen2.5-14b-instruct-1m
[vllm-swift] kv-cache: 4.8 GB
[vllm-swift] generation: 78.2 tok/sec
"""


def test_vllm_swift_parses_decode_tps_and_kv_size():
    e = VllmSwiftEngine(binary="/fake/vllm-swift")
    with patch("tqkit.engines.vllm_swift.subprocess.run",
               return_value=_fake_proc(stderr=VLLM_SWIFT_LOG)):
        result = e.run(model="m", prompt="p", ctx_tokens=32768,
                       layout="tq+asym")
    assert result.backend == "vllm-swift"
    assert result.decode_tps == pytest.approx(78.2)
    assert result.kv_cache_bytes == int(4.8 * (1000 ** 3))


def test_vllm_swift_run_raises_when_not_installed():
    """Mock shutil.which so the test passes regardless of host install state."""
    with patch("tqkit.engines.vllm_swift.shutil.which", return_value=None):
        e = VllmSwiftEngine()
    with pytest.raises(EngineNotInstalled):
        e.run(model="m", prompt="p", ctx_tokens=1024)


def test_vllm_swift_unsupported_layout():
    e = VllmSwiftEngine(binary="/fake/vllm-swift")
    with pytest.raises(ValueError, match="layout"):
        e.run(model="m", prompt="p", ctx_tokens=1024, layout="bogus")


# --- longctx (retrieval-as-savings) ---

def test_longctx_supports_canonical_layouts():
    """longctx is layout-agnostic but accepts the canonical names so the
    bench dispatcher can iterate uniformly."""
    e = LongctxEngine()
    assert e.supports("fp16")
    assert e.supports("tq+asym")
    assert not e.supports("not-a-layout")


def test_longctx_run_raises_when_not_installed():
    """If the longctx package isn't importable, run() should raise cleanly."""
    e = LongctxEngine()
    with patch.object(e, "is_installed", return_value=False):
        with pytest.raises(EngineNotInstalled):
            e.run(model="m", prompt="p", ctx_tokens=1024)


def test_longctx_run_dispatches_to_LongCtxClient():
    """Verify the engine wires through to longctx.LongCtxClient.ask and
    materializes a RunResult with retrieval-style metadata."""
    e = LongctxEngine(server="http://fake:5050/v1/chat/completions")
    fake_response = type("R", (), {
        "content": "answer", "retrieved_indices": [0, 2, 5],
        "prompt_tokens": 3000, "completion_tokens": 50, "latency_s": 0.3,
    })()
    fake_client_cls = type("C", (), {
        "__init__": lambda self, **kw: None,
        "ask": lambda self, **kw: fake_response,
    })
    fake_module = type("M", (), {"LongCtxClient": fake_client_cls})
    with patch.object(e, "is_installed", return_value=True), \
         patch.dict("sys.modules", {"longctx": fake_module}):
        result = e.run(
            model="qwen25-32b", prompt="what about q3 revenue?",
            ctx_tokens=1_048_576,
            candidates=["c0", "c1", "c2"], top_k=8,
            layout="tq+asym",
        )
    assert result.backend == "longctx"
    assert result.layout == "longctx+tq+asym"
    assert result.kv_cache_bytes == 3000 * 192 * 1024
    assert "retrieval (not compression)" in result.extra["savings_strategy"]
    assert result.extra["retrieved_token_count"] == "3000"
