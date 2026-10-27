"""Accessible CLI with real execution, checked errors, and portable artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .backend import Hardware
from .benchmark import run_benchmarks
from .importer import import_onnx
from .pipeline import Run, compile_model
from .reports import render_report


def print_run(label: str, run: Run) -> None:
    m = run.metrics
    c = run.changes
    print(f"\n{label} · {run.model} · {run.program.hardware.architecture}")
    print("  [PASS] ONNX validation / graph import / IR verification")
    print(
        f"  [PASS] Optimization: {c['original_operations']} → {c['final_operations']} ops; "
        f"{c['dead_operations_removed']} dead; {c['fusion_patterns']} fused"
    )
    print(f"  [PASS] Hardware lowering: {len(run.program.instructions):,} instructions")
    print("  [PASS] Native C++ simulation / numerical verification against ONNX Runtime")
    print(
        f"  Cycles {m['cycles']:,}  |  Latency {m['latency_ns']:,.0f} ns  |  "
        f"Off-chip {m['offchip_bytes']:,} B  |  SRAM peak {m['peak_sram_bytes']:,} B"
    )
    print(
        f"  Compute {m['compute_utilization']:.1%}  |  DMA {m['dma_utilization']:.1%}  |  "
        f"PIM {m['pim_utilization']:.1%}  |  Pressure: {m['bottleneck']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NPUForge — ONNX-to-NPU compiler & hardware-aware simulator"
    )
    parser.add_argument("--version", action="version", version=f"NPUForge {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "benchmark"):
        command = commands.add_parser(name, help="run both demos and all six comparisons")
        command.add_argument("--output", type=Path, default=Path("output"))
        command.add_argument("--config", type=Path, help="JSON hardware configuration")
    compile_parser = commands.add_parser("compile", help="compile, simulate, verify and report")
    compile_parser.add_argument("model", type=Path)
    compile_parser.add_argument("--output", type=Path, default=Path("output"))
    compile_parser.add_argument("--config", type=Path)
    compile_parser.add_argument("--arch", choices=["conventional", "pim"])
    compile_parser.add_argument("--no-opt", action="store_true")
    compile_parser.add_argument(
        "--compare", action="store_true", help="also run the other architecture"
    )
    inspect_parser = commands.add_parser("inspect", help="validate and print imported IR")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            graph = import_onnx(args.model)
            print(graph.to_json() if args.json else str(graph))
            return 0
        config = Hardware.load(args.config) if args.config else Hardware()
        print("NPUForge\nONNX-to-NPU Compiler & Hardware Simulator")
        if args.command in {"demo", "benchmark"}:
            runs = run_benchmarks(args.output, config)
        else:
            if args.arch:
                config = config.alternative(args.arch)
            run = compile_model(args.model, config, not args.no_opt)
            run.write(args.output / config.architecture)
            runs = [(config.architecture, run)]
            if args.compare:
                other = "pim" if config.architecture == "conventional" else "conventional"
                alternative = compile_model(args.model, config.alternative(other), not args.no_opt)
                alternative.write(args.output / other)
                runs.append((other, alternative))
        for label, run in runs:
            print_run(label, run)
        report = render_report(runs, args.output / "report.html")
        print(f"\nReport: {report}\nArtifacts: {args.output}")
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(
            f"npuforge: {exc}\nCheck model support, file paths and hardware capacities. "
            "See README.md and docs/architecture.md.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
