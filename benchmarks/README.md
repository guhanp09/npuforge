# Reproducible experiments

Run `npuforge benchmark --output output/benchmarks` after installing the package.
It compiles every case, invokes the C++ engine, verifies NumPy outputs against ONNX
Runtime, writes full IR/instruction/trace artifacts, and generates JSON, Markdown
and an HTML report. `python scripts/refresh_artifacts.py` refreshes the checked-in
reference tables and SVG figures from these same executions.

Workload: batch 16, input 32, hidden 24, MLP output 12. MatMul is `(16,32) @ (32,24)`.
Weights: NumPy default_rng seed 11, normal distribution scaled by 0.2; runtime inputs:
seed 7, standard normal. Default configuration is `configs/conventional.json`, with
only architecture changed for PIM. The alternate MatMul tile is `(8,8,8)`; requested
default is `(16,16,16)`. All six cases are retained, including the slower small tile.

`results.json` contains hardware parameters, transformations, correctness errors,
per-operation timings, and complete performance metrics. `environment.json` records
the actual reference host and package versions. Cycle/byte counts are deterministic
for this model and configuration. Tiny numerical error differences across BLAS
implementations are permitted by the documented tolerance.

The clock is 1 GHz, so the numbers in cycles and nanoseconds happen to match.
These are abstract simulated accelerator estimates, not host timings or real-chip
measurements. Initial data is already in DRAM; no setup or model-loading time is
included. PIM-local reads consume bandwidth and cycles and are counted separately
from off-chip DMA traffic. See [the hardware specification](../docs/architecture.md).

## Gemm/Relu fusion comparison

Run `python examples/compare_gemm_fusion.py` to generate four additional cases and
`output/gemm-fusion/report.html`. The workload is a `(16,32)` input times stored
`(24,32)` weights with transB=1, alpha=0.75, beta=0.5, vector bias and trailing Relu.
It uses the same seeds and default hardware as the original benchmark set.

`gemm_fusion_results.json` and `gemm_fusion_results.md` contain the actual reference
results. `python scripts/refresh_artifacts.py` regenerates both benchmark sets and
reports. Linux validation independently compares all four additional native metric
dictionaries with this reference. The original six benchmark results remain intact.
