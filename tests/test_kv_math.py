"""Tests for KV cache math + report/table CLI commands."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr

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


def test_tq_asym_savings_is_625_pct():
    """K=q8_0 (1B) + V=turbo4 (0.5B) vs FP16 (4B) = 62.5% savings."""
    arch = MODELS["qwen2.5-14b-instruct-1m"]
    fp16 = kv_bytes_total(arch, "fp16", 1024)
    tq = kv_bytes_total(arch, "tq+asym", 1024)
    assert savings_pct(fp16, tq) == pytest.approx(62.5)


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
    assert "62.5%" in out
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
