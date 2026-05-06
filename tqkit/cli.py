"""tqkit CLI entry point. Provides the `tq` command."""
from __future__ import annotations

import argparse
import sys

from tqkit import __version__
from tqkit.backends import detect_all
from tqkit.kv_math import (
    LAYOUT_BYTES_PER_ELEM,
    MODELS,
    fmt_bytes,
    kv_bytes_per_token,
    kv_bytes_total,
    savings_pct,
)


def cmd_backends(args: argparse.Namespace) -> int:
    rows = detect_all()
    print(f"{'backend':<14} {'available':<10} {'version':<30} notes")
    print("-" * 80)
    for r in rows:
        print(
            f"{r.name:<14} {('yes' if r.available else 'no'):<10} "
            f"{(r.version or '-'):<30} {r.notes}"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    print("`tq bench` — canonical KV-savings benchmark", file=sys.stderr)
    print("(coming in v0.2.0; see TheTom/tqkit GitHub roadmap)", file=sys.stderr)
    return 0


def _parse_ctx(s: str) -> int:
    """Parse context strings like '32K', '1M', '128000'."""
    s = s.strip().upper()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def cmd_report(args: argparse.Namespace) -> int:
    """Print theoretical KV-cache size for a model + ctx + layout.

    Computed from pinned HF architecture (head_dim, num_kv_heads, num_layers).
    Compares baseline FP16 to the requested layout. Used for reproducibility
    framing and to populate the canonical bench table without needing each
    engine to expose runtime KV-cache instrumentation.
    """
    if args.model not in MODELS:
        print(f"unknown model '{args.model}'", file=sys.stderr)
        print(f"available: {', '.join(sorted(MODELS))}", file=sys.stderr)
        return 1
    arch = MODELS[args.model]
    ctx = _parse_ctx(args.ctx)

    fp16_total = kv_bytes_total(arch, "fp16", ctx)
    target_total = kv_bytes_total(arch, args.layout, ctx)
    fp16_per_tok = kv_bytes_per_token(arch, "fp16")
    target_per_tok = kv_bytes_per_token(arch, args.layout)
    sav = savings_pct(fp16_total, target_total)

    print(f"[KV cache] model: {arch.name}")
    print(f"[KV cache] arch: layers={arch.num_layers} "
          f"kv_heads={arch.num_kv_heads} head_dim={arch.head_dim}")
    print(f"[KV cache] layout: {args.layout}")
    print(f"[KV cache] per-token: {fmt_bytes(target_per_tok)} "
          f"(vs {fmt_bytes(fp16_per_tok)} FP16)")
    print(f"[KV cache] total @ {args.ctx} ctx: {fmt_bytes(target_total)} "
          f"(vs {fmt_bytes(fp16_total)} FP16, {sav:.1f}% savings)")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    """Print the unified KV-savings table for one model across layouts."""
    if args.model not in MODELS:
        print(f"unknown model '{args.model}'", file=sys.stderr)
        return 1
    arch = MODELS[args.model]
    layouts = args.layouts or ["fp16", "tq+sym", "tq+asym"]
    ctxs = [_parse_ctx(c) for c in (args.ctxs or ["8K", "32K", "64K", "1M"])]

    print(f"# {arch.name} — KV cache size by layout × context")
    print()
    header_ctx = " | ".join(f"{c // 1024}K" if c < 1024 * 1024
                            else f"{c // (1024 * 1024)}M"
                            for c in ctxs)
    print(f"| layout | per-token | {header_ctx} | savings vs FP16 |")
    print(f"| ------ | --------- | "
          + " | ".join("---" for _ in ctxs) + " | --- |")
    fp16_max = kv_bytes_total(arch, "fp16", max(ctxs))
    for layout in layouts:
        per_tok = kv_bytes_per_token(arch, layout)
        cells = [fmt_bytes(kv_bytes_total(arch, layout, c)) for c in ctxs]
        target_max = kv_bytes_total(arch, layout, max(ctxs))
        sav = savings_pct(fp16_max, target_max)
        sav_str = "—" if layout == "fp16" else f"{sav:.0f}%"
        print(f"| {layout} | {fmt_bytes(per_tok)} | "
              + " | ".join(cells) + f" | {sav_str} |")
    return 0


def cmd_integrate(args: argparse.Namespace) -> int:
    recipes = {
        "llama-cpp": (
            "git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama.cpp\n"
            "cd llama.cpp && cmake -B build && cmake --build build -j\n"
            "./build/bin/llama-cli -m <model> --ctk q8_0 --ctv turbo4 ..."
        ),
        "vllm-cuda": (
            "git clone -b feature/turboquant-kv-cache https://github.com/TheTom/vllm\n"
            "cd vllm && pip install -e ."
        ),
        "vllm-amd": (
            "docker pull thetom/vllm-turboquant:rocm-7.2  # coming v0.2.0\n"
            "or build from TheTom/vllm@feature/turboquant-amd"
        ),
        "mlx-swift": (
            "swift package add TheTom/mlx-swift-lm@feature/turboquant-plus\n"
            "or use Pal iOS app with TQ+ toggle enabled"
        ),
        "vllm-swift": (
            "brew install thetom/vllm-swift\n"
            "vllm-swift serve <model> --kv-layout tq+asym  # flag coming v0.2.0"
        ),
    }
    if args.backend not in recipes:
        print(f"unknown backend '{args.backend}'", file=sys.stderr)
        print(f"available: {', '.join(sorted(recipes))}", file=sys.stderr)
        return 1
    print(recipes[args.backend])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tq",
        description="TurboQuant+ KV-cache toolkit (https://github.com/TheTom/tqkit)",
    )
    p.add_argument("--version", action="version",
                   version=f"tqkit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub_backends = sub.add_parser("backends",
                                  help="autodetect installed inference engines")
    sub_backends.set_defaults(func=cmd_backends)

    sub_bench = sub.add_parser("bench",
                               help="run canonical KV-savings benchmark")
    sub_bench.set_defaults(func=cmd_bench)

    sub_report = sub.add_parser(
        "report",
        help="print theoretical KV-cache size for a model + ctx + layout",
    )
    sub_report.add_argument(
        "--model", required=True,
        help=f"one of: {', '.join(sorted(MODELS))}",
    )
    sub_report.add_argument(
        "--ctx", default="32K",
        help="context length (e.g. 8K, 32K, 1M, or raw int). default 32K",
    )
    sub_report.add_argument(
        "--layout", default="tq+asym",
        choices=sorted(LAYOUT_BYTES_PER_ELEM),
        help="KV cache layout. default tq+asym",
    )
    sub_report.set_defaults(func=cmd_report)

    sub_table = sub.add_parser(
        "table",
        help="print KV-savings table (one model across layouts × ctxs)",
    )
    sub_table.add_argument(
        "--model", required=True,
        help=f"one of: {', '.join(sorted(MODELS))}",
    )
    sub_table.add_argument(
        "--layouts", nargs="+",
        help="layouts to compare. default: fp16 tq+sym tq+asym",
    )
    sub_table.add_argument(
        "--ctxs", nargs="+",
        help="context lengths. default: 8K 32K 64K 1M",
    )
    sub_table.set_defaults(func=cmd_table)

    sub_integrate = sub.add_parser(
        "integrate",
        help="print install + serve recipe for one engine",
    )
    sub_integrate.add_argument(
        "backend",
        choices=["llama-cpp", "vllm-cuda", "vllm-amd", "mlx-swift", "vllm-swift"],
    )
    sub_integrate.set_defaults(func=cmd_integrate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
