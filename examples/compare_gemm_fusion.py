"""Generate, verify and compare Gemm/Relu fusion on both simulated architectures."""

import argparse
from pathlib import Path

from npuforge.benchmark import benchmark_table, run_fusion_benchmarks
from npuforge.reports import render_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/gemm-fusion"))
args = parser.parse_args()
runs = run_fusion_benchmarks(args.output)
report = render_report(runs, args.output / "report.html")
print(benchmark_table(runs))
print(f"All four cases verified against ONNX Runtime.\nReport: {report}")
