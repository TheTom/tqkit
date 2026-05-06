"""tqkit CLI entry point. Provides the `tq` command."""
from __future__ import annotations

import argparse
import sys

from tqkit import __version__
from tqkit.backends import detect_all


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


def cmd_report(args: argparse.Namespace) -> int:
    print("`tq report` — KV-cache layout reporter", file=sys.stderr)
    print("(coming in v0.2.0; ports KV reporter from each engine)", file=sys.stderr)
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

    sub_report = sub.add_parser("report",
                                help="print the most recent KV-cache layout report")
    sub_report.set_defaults(func=cmd_report)

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
