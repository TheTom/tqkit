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
    """Run the canonical KV-savings benchmark across detected backends.

    For each (backend × layout) cell:
      1. Dispatch to the matching tqkit.engines.<backend> bridge.
      2. Capture the engine's normalized RunResult.
      3. Tally into a markdown summary table.

    Skips backends that aren't installed (with a note) and engines that
    don't support the requested layout.
    """
    from tqkit.engines import ENGINES, get_engine, EngineNotInstalled

    backends = args.backends or list(ENGINES)
    layouts = args.layouts or ["fp16", "tq+asym"]
    rows: list[tuple[str, str, str | None, str | None]] = []

    prompt = args.prompt or "Hello, world. Briefly summarize this prompt."
    ctx = _parse_ctx(args.ctx)

    for be_name in backends:
        try:
            engine = get_engine(be_name)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        if not engine.is_installed():
            print(f"[skip] {be_name}: not installed", file=sys.stderr)
            continue
        for layout in layouts:
            if not engine.supports(layout):
                rows.append((be_name, layout, None,
                             f"layout not supported by {be_name}"))
                continue
            try:
                result = engine.run(
                    model=args.model, prompt=prompt, ctx_tokens=ctx,
                    layout=layout, max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            except EngineNotInstalled as e:
                rows.append((be_name, layout, None, str(e)))
                continue
            kv_str = ("?" if result.kv_cache_bytes is None
                      else _gb(result.kv_cache_bytes))
            tps_str = (f"{result.decode_tps:.1f}" if result.decode_tps
                       else "?")
            rows.append((be_name, layout,
                         f"kv={kv_str} tps={tps_str}", None))

    print(f"# tqkit bench — model={args.model} ctx={args.ctx}")
    print()
    print("| backend | layout | result | note |")
    print("| ------- | ------ | ------ | ---- |")
    for be, layout, result, note in rows:
        print(f"| {be} | {layout} | {result or '—'} | {note or ''} |")
    return 0


def _gb(bytes_count: int) -> str:
    return f"{bytes_count / (1024 ** 3):.2f} GB"


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


def cmd_config(args: argparse.Namespace) -> int:
    """Print the canonical benchmark config (pinned model + prompt + expected results)."""
    import importlib.resources as ir
    try:
        text = ir.files("tqkit").joinpath("canonical_bench.yml").read_text()
    except (FileNotFoundError, AttributeError):
        # development install fallback
        from pathlib import Path
        text = (Path(__file__).parent / "canonical_bench.yml").read_text()
    print(text)
    return 0


def cmd_compare_strategies(args: argparse.Namespace) -> int:
    """Compare KV cache savings strategies side by side: compress vs
    avoid vs combined.

    The unified narrative: TurboQuant+ shrinks the KV cache (compression);
    longctx avoids it entirely by retrieving the relevant chunks
    (avoidance). They combine multiplicatively. The most efficient KV
    cache is the one you never allocate.
    """
    if args.model not in MODELS:
        print(f"unknown model '{args.model}'", file=sys.stderr)
        return 1
    arch = MODELS[args.model]
    ctx = _parse_ctx(args.ctx)
    retrieved = args.retrieved_tokens

    fp16_bytes = kv_bytes_total(arch, "fp16", ctx)
    tq_bytes = kv_bytes_total(arch, "tq+asym", ctx)
    longctx_bytes = kv_bytes_total(arch, "fp16", retrieved)
    combined_bytes = kv_bytes_total(arch, "tq+asym", retrieved)

    rows = [
        ("baseline (FP16, full ctx)", fp16_bytes,
         "no compression, no retrieval"),
        ("TQ+ asym (compress, full ctx)", tq_bytes,
         "K=FP8, V=4bit + metadata"),
        ("longctx (FP16, top-K only)", longctx_bytes,
         "retrieve relevant chunks, no compression"),
        ("longctx + TQ+ asym (combined)", combined_bytes,
         "the most efficient KV cache is the one you never allocate"),
    ]

    print(f"# {arch.name} — KV cache savings strategies @ {args.ctx} ctx")
    print(f"# (assuming retrieval keeps {retrieved:,} tokens)")
    print()
    print("| strategy | KV cache | savings vs baseline | notes |")
    print("| -------- | -------- | ------------------- | ----- |")
    for name, b, note in rows:
        sav = savings_pct(fp16_bytes, b)
        sav_str = "—" if b == fp16_bytes else f"{sav:.2f}%"
        print(f"| {name} | {fmt_bytes(b)} | {sav_str} | {note} |")
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
            "cd vllm && pip install -e .\n"
            "vllm serve <model> --kv-cache-dtype turboquant_k8v4   # asym\n"
            "vllm serve <model> --kv-cache-dtype turboquant_4bit_nc # sym"
        ),
        "vllm-amd": (
            "docker pull thetom/vllm-turboquant:rocm-7.2\n"
            "or build from TheTom/vllm@feature/turboquant-amd-noautotune\n"
            "vllm serve <model> --kv-cache-dtype turboquant_k8v4   # asym, headline"
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
    sub_bench.add_argument("--model", required=True,
                           help="model path or HF repo (engine-specific)")
    sub_bench.add_argument("--ctx", default="32K",
                           help="context length (default 32K)")
    sub_bench.add_argument("--backends", nargs="+",
                           help="subset of backends to run; default: all detected")
    sub_bench.add_argument("--layouts", nargs="+",
                           help="layouts to compare; default: fp16 tq+asym")
    sub_bench.add_argument("--prompt", default=None,
                           help="prompt text; default: short canned probe")
    sub_bench.add_argument("--max-tokens", type=int, default=64)
    sub_bench.add_argument("--timeout", type=int, default=600)
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

    sub_config = sub.add_parser(
        "config",
        help="print the canonical benchmark YAML (pinned model + expected results)",
    )
    sub_config.set_defaults(func=cmd_config)

    sub_compare = sub.add_parser(
        "compare-strategies",
        help="compare compress (TQ+) vs avoid (longctx) vs combined",
    )
    sub_compare.add_argument(
        "--model", required=True,
        help=f"one of: {', '.join(sorted(MODELS))}",
    )
    sub_compare.add_argument(
        "--ctx", default="1M",
        help="full-context size the user *would* serve. default 1M",
    )
    sub_compare.add_argument(
        "--retrieved-tokens", type=int, default=3000,
        help="tokens longctx retrieves and feeds to the model. default 3000",
    )
    sub_compare.set_defaults(func=cmd_compare_strategies)

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
