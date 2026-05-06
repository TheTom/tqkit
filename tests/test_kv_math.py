"""Tests for KV cache math + report/table CLI commands."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import pytest

from tqkit.kv_math import (
    LAYOUT_BYTES_PER_ELEM,
    MODELS,
    fmt_bytes,
    kv_bytes_per_token,
    kv_bytes_total,
    savings_pct,
)
from tqkit.cli import _parse_ctx, main


def test_qwen14b_fp16_per_token_matches_reference():
    """Sanity: Qwen2.5-14B-1M @ FP16 = 192 KB/token. (48 layers × 8 kv_heads × 128 head_dim × 4 bytes)"""
    arch = MODELS["qwen2.5-14b-instruct-1m"]
    bytes_per_tok = kv_bytes_per_token(arch, "fp16")
    assert bytes_per_tok == 48 * 8 * 128 * 4
    assert bytes_per_tok == 192 * 1024


def test_tq_asym_savings_matches_verified_bpe():
    """tq+asym (K=FP8 1.0 B, V=4bit+metadata 0.5313 B) vs FP16 (4 B sum) =
    1 - (1.0 + 0.5313) / 4 = 61.72% savings. Verified against
    TheTom/vllm@feature/turboquant-kv-cache turboquant_k8v4 preset, including
    fp16 scale+zero metadata on V."""
    arch = MODELS["qwen2.5-14b-instruct-1m"]
    fp16 = kv_bytes_total(arch, "fp16", 1024)
    tq = kv_bytes_total(arch, "tq+asym", 1024)
    assert savings_pct(fp16, tq) == pytest.approx(61.72, abs=0.01)


def test_boundary_skip_keeps_two_layers_fp16_at_each_end():
    """The vllm fork keeps first/last 2 attention layers at FP16. Effective
    KV cache is bigger than the all-quantized math suggests."""
    from tqkit.kv_math import kv_bytes_total_with_boundary_skip

    arch = MODELS["qwen2.5-14b-instruct-1m"]  # 48 attention layers
    naive = kv_bytes_total(arch, "tq+asym", ctx=1024)
    realistic = kv_bytes_total_with_boundary_skip(
        arch, "tq+asym", ctx=1024, n_skip=2,
    )
    # Realistic is bigger: 4 FP16 layers cost more per layer than 44 TQ+ layers.
    assert realistic > naive
    # Sanity: skip=0 returns the naive value
    no_skip = kv_bytes_total_with_boundary_skip(
        arch, "tq+asym", ctx=1024, n_skip=0,
    )
    assert no_skip == pytest.approx(naive)
    # FP16 layout doesn't get the skip treatment (no quantized layer to skip)
    fp16 = kv_bytes_total(arch, "fp16", ctx=1024)
    fp16_skip = kv_bytes_total_with_boundary_skip(
        arch, "fp16", ctx=1024, n_skip=2,
    )
    assert fp16_skip == fp16


def test_unknown_layout_raises():
    arch = MODELS["qwen2.5-14b-instruct-1m"]
    with pytest.raises(ValueError, match="unknown layout"):
        kv_bytes_per_token(arch, "not-a-layout")


def test_kv_bytes_total_scales_linearly_with_ctx():
    arch = MODELS["qwen2.5-7b-instruct"]
    a = kv_bytes_total(arch, "fp16", 1000)
    b = kv_bytes_total(arch, "fp16", 4000)
    assert b == 4 * a


def test_fmt_bytes_units():
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(2048) == "2.0 KB"
    assert fmt_bytes(2 * 1024 * 1024) == "2.0 MB"
    assert fmt_bytes(3 * 1024 ** 3) == "3.0 GB"
    assert fmt_bytes(1.5 * 1024 ** 4) == "1.5 TB"


def test_savings_pct_handles_zero():
    assert savings_pct(0, 100) == 0.0
    assert savings_pct(100, 25) == 75.0


def test_parse_ctx_handles_k_m_and_raw():
    assert _parse_ctx("8K") == 8 * 1024
    assert _parse_ctx("32k") == 32 * 1024
    assert _parse_ctx("1M") == 1024 * 1024
    assert _parse_ctx("131072") == 131072


def test_all_known_models_have_consistent_arch():
    """Every pinned model should produce a positive per-token KV cost."""
    for name, arch in MODELS.items():
        assert arch.num_layers > 0, name
        assert arch.num_kv_heads > 0, name
        assert arch.head_dim > 0, name
        assert kv_bytes_per_token(arch, "fp16") > 0, name


def test_all_layouts_are_smaller_than_fp16_or_equal():
    arch = MODELS["qwen2.5-14b-instruct-1m"]
    fp16 = kv_bytes_per_token(arch, "fp16")
    for layout in LAYOUT_BYTES_PER_ELEM:
        assert kv_bytes_per_token(arch, layout) <= fp16, layout


def test_cli_report_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["report", "--model", "qwen2.5-14b-instruct-1m",
                   "--ctx", "32K", "--layout", "tq+asym"])
    assert rc == 0
    out = buf.getvalue()
    assert "Qwen2.5-14B-Instruct-1M" in out
    # Verified bpe = 1.0/0.5313 → 61.7% savings, not the old 62.5%
    assert "61.7%" in out or "61.72%" in out
    assert "tq+asym" in out


def test_cli_report_unknown_model_returns_1():
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        rc = main(["report", "--model", "fake-model",
                   "--ctx", "8K", "--layout", "fp16"])
    assert rc == 1
    assert "unknown model" in buf_err.getvalue()


def test_cli_table_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["table", "--model", "qwen2.5-14b-instruct-1m"])
    assert rc == 0
    out = buf.getvalue()
    assert "fp16" in out
    assert "tq+asym" in out
    assert "192.0 KB" in out  # FP16 per-token


def test_cli_table_with_unknown_layout_errors():
    """Bogus layout names raise from the math layer."""
    err = io.StringIO()
    with redirect_stderr(err):
        with pytest.raises((ValueError, SystemExit)):
            main([
                "table", "--model", "qwen2.5-14b-instruct-1m",
                "--layouts", "fp16", "not-a-real-layout",
            ])


def test_cli_table_custom_layouts_and_ctxs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "table", "--model", "qwen2.5-7b-instruct",
            "--layouts", "fp16", "tq+asym",
            "--ctxs", "8K", "32K",
        ])
    assert rc == 0
    out = buf.getvalue()
    # 8K and 32K columns present, others absent
    assert "8K" in out and "32K" in out
    assert "1M" not in out


def test_cli_backends_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["backends"])
    assert rc == 0
    assert "llama.cpp" in buf.getvalue()


def test_cli_integrate_known_backend():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["integrate", "vllm-amd"])
    assert rc == 0
    assert "feature/turboquant-amd" in buf.getvalue()


def test_cli_integrate_unknown_backend():
    """argparse rejects unknown choice with SystemExit(2) before main returns."""
    with pytest.raises(SystemExit) as exc:
        main(["integrate", "fake"])
    assert exc.value.code == 2


def test_cli_config_prints_yaml():
    """`tq config` emits the pinned canonical_bench.yml."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config"])
    assert rc == 0
    out = buf.getvalue()
    # Spot-checks: schema markers any caller of the bench would care about
    assert "version: 1" in out
    assert "qwen2.5-14b-instruct-1m" in out
    assert "killer_demo" in out
    assert "vllm_amd" in out


def test_cli_bench_runs_with_no_installed_backends():
    """`tq bench` should print an empty (header-only) table without crashing
    when no engines are installed on the host."""
    with patch("tqkit.engines.LlamaCppEngine") as mll, \
         patch("tqkit.engines.VllmCudaEngine") as mvc, \
         patch("tqkit.engines.VllmAmdEngine") as mva, \
         patch("tqkit.engines.MlxSwiftEngine") as mms, \
         patch("tqkit.engines.VllmSwiftEngine") as mvs:
        for m in (mll, mvc, mva, mms, mvs):
            m.return_value.is_installed.return_value = False
            m.return_value.name = "fake"
        buf = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = main([
                "bench", "--model", "fake-model",
                "--ctx", "8K", "--layouts", "fp16",
            ])
    assert rc == 0
    assert "tqkit bench" in buf.getvalue()
    assert "| backend |" in buf.getvalue()


def test_cli_bench_dispatches_to_engine_run():
    """When an engine is installed, `tq bench` calls its run() method
    once per layout. Patches `get_engine` so the dispatcher returns our mock
    regardless of host install state."""
    from tqkit.engines.base import RunResult
    from unittest.mock import MagicMock

    fake_result = RunResult(
        backend="vllm-cuda", model="m", layout="tq+asym",
        ctx_tokens=8192, kv_cache_bytes=2 * (1024 ** 3),
        decode_tps=88.5,
    )
    fake_engine = MagicMock()
    fake_engine.is_installed.return_value = True
    fake_engine.supports.return_value = True
    fake_engine.run.return_value = fake_result
    fake_engine.name = "vllm-cuda"

    # cmd_bench does `from tqkit.engines import get_engine` at call time,
    # so patch the source module rather than the cli namespace.
    with patch("tqkit.engines.get_engine", return_value=fake_engine):
        buf = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = main([
                "bench", "--model", "fake-model",
                "--backends", "vllm-cuda",
                "--layouts", "tq+asym",
                "--ctx", "8K",
            ])
    assert rc == 0
    assert fake_engine.run.called
    out = buf.getvalue()
    assert "vllm-cuda" in out
    assert "kv=" in out
    assert "tps=" in out


def test_canonical_bench_yaml_parses_and_has_required_keys():
    """The shipped canonical_bench.yml is well-formed and has the expected schema."""
    yaml = pytest.importorskip("yaml")
    from pathlib import Path
    text = (Path(__file__).parent.parent / "tqkit" / "canonical_bench.yml").read_text()
    data = yaml.safe_load(text)
    assert data["version"] == 1
    assert data["model"]["name"] == "qwen2.5-14b-instruct-1m"
    assert "tq+asym" in data["layouts"]
    assert {"llama_cpp", "vllm_cuda", "vllm_amd", "mlx_swift", "vllm_swift"}.issubset(
        set(data["backends"])
    )
    assert data["killer_demo"]["ctx_tokens"] == 1048576
