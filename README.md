# NPUForge

**An ONNX-to-NPU compiler and native C++17 hardware simulator.** Import real static
FP32 models, optimize a typed tensor graph, emit scratchpad-bounded instructions,
and study compute, DMA, and memory contention on a documented hypothetical NPU.

![Compiler and simulator architecture](docs/figures/architecture.svg)

The complete pipeline runs on a normal CPU. NumPy implements numerical semantics;
ONNX Runtime independently checks results. C++ schedules the generated instruction
DAG and calculates performance. No accelerator, account, downloaded model or cloud
service is required.

## Quick start

Prerequisites: Python **3.11+** and a C++17 compiler. On macOS, install Apple's
Command Line Tools (`xcode-select --install`) if needed. On Ubuntu, install
`python3-venv` and `build-essential`. CMake and pybind11 are provisioned automatically
in pip's isolated native build.

```bash
git clone https://github.com/guhanp09/npuforge.git
cd npuforge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
npuforge demo
```

Open **`output/report.html`** in any browser. It is fully self-contained. The demo
creates two real ONNX models, verifies both, and runs all six compiler/architecture
comparisons. Inspectable IR, instruction and native trace JSON accompany the report.

Actual excerpt from the default run:

```text
mlp_optimized · demo_mlp.onnx · conventional
  [PASS] ONNX validation / graph import / IR verification
  [PASS] Optimization: 5 → 3 ops; 0 dead; 1 fused
  [PASS] Hardware lowering: 35 instructions
  [PASS] Native C++ simulation / numerical verification against ONNX Runtime
  Cycles 945  |  Latency 945 ns  |  Off-chip 13,840 B  |  SRAM peak 4,096 B
```

![Screenshot of the generated engineering report](docs/figures/report.png)

## What is implemented

- **ONNX importer:** validation, shape inference, FP32 initializers, public inputs
  and outputs, dependencies, and explicit rejection of unsupported models.
- **Compiler IR:** tensor shapes/types, operations and attributes, graph invariants,
  stable topological order, producer/consumer maps, readable text, deterministic
  JSON serialization and round trips.
- **Real optimization:** backwards dead computation elimination and safe
  `MatMul → Add → Relu` and `Gemm → Relu` fusion, preserving observable/shared intermediates.
- **Hardware backend:** M/N/K tiling, boundary handling, accumulation, explicit
  DMA and SRAM allocation/release, conventional and memory-local matrix mappings.
- **Native simulator:** deterministic event scheduling, dependency checks, SRAM
  pressure, compute/DMA/PIM contention, a shared DRAM port, execution traces,
  modeled latency, movement, utilization and resource pressure analysis.
- **Verification and presentation:** independent NumPy execution, ONNX Runtime
  comparisons, six reproducible benchmarks, an offline HTML report with graph and
  timeline SVGs, unit/integration/stress tests and native timing tests.

## Run your own model

```bash
npuforge inspect output/models/demo_mlp.onnx
npuforge inspect output/models/demo_mlp.onnx --json
npuforge compile output/models/demo_mlp.onnx --compare --output output/custom
npuforge compile output/models/demo_mlp.onnx --no-opt --output output/baseline
npuforge compile output/models/demo_matmul.onnx --config configs/pim.json --output output/pim
npuforge benchmark --output output/benchmarks
```

`compile` always imports, compiles, simulates, verifies, and writes a report. Use
`--arch conventional|pim` to override the configuration's architecture. Hardware
configurations are plain JSON, with validated capacities, rates, setup costs,
clock and tile bounds. Errors return a nonzero exit status with an actionable
message. [Examples](examples/README.md) show generating and inspecting models.

## Compiler and simulator decisions

The importer accepts default-domain ONNX opsets **13–18** with positive static
shapes. `MatMul` supports **rank-2 `(M,K) @ (K,N)`**. `Gemm` supports rank-2
`alpha * op(A) @ op(B) + beta * C`, both transpose flags, finite FP32 coefficients,
and an optional bias C that broadcasts into `(M,N)`. `Add` supports modern ONNX
multidirectional broadcasting, including scalars; `Relu` supports static tensors.
Only finite FP32 initializers are accepted. Unknown operations and attributes,
custom domains, sparse weights and unresolved shapes are rejected.

Generate and compile a dense layer:

```bash
python examples/generate_models.py --output output/models
npuforge compile output/models/demo_gemm.onnx --compare --output output/gemm
```

The example uses transposed weight storage, alpha 0.75 and beta 0.5. It verifies
against ONNX Runtime and reports 651 conventional / 547 PIM simulated cycles with
the default hardware. Scaling and bias arithmetic cost cycles on the final K tile;
transposed addressing uses the model's existing ideal DMA/local gather abstraction.
Unoptimized Gemm remains a distinct IR operation. Eligible
`Gemm → Relu` chains become `FusedGemmRelu`; shared or public Gemm results remain
observable. The activation executes only after the final reduction, scaling and
bias addition, with its arithmetic counted in the native simulation.

Compare fusion on both architectures in one report:

```bash
python examples/compare_gemm_fusion.py
```

Open `output/gemm-fusion/report.html`. On the deterministic example, fusion reduces
conventional cycles **791 → 675** and PIM cycles **687 → 571**, avoiding **3,072 B**
of off-chip movement in either mapping. The complete four-case results, including
unoptimized baselines, are in [the fusion benchmark](benchmarks/gemm_fusion_results.md).

The IR is independent of ONNX protobufs. Fusion removes intermediate materialization
and executes the bias/activation epilogue on the final reduction tile.
The lowerer reserves all input, weight,
accumulator and optional bias staging before issuing a tile, and reduces tile sizes
until they fit. It spills operation outputs to DRAM and conservatively uses a single
tile workspace. There is no global buffer or scheduling optimizer.

The C++ simulator can overlap independent compute and DMA instructions, enforcing
exclusive resources and live SRAM capacity. Total cycles are the **last completion
timestamp**, so overlapping work is not double-counted. Busy cycles divided by that
makespan give per-resource utilization. Allocation deadlocks, invalid dependencies,
unknown instructions, unsafe frees and oversized working sets fail clearly.

In PIM mode, the right matrix operand stays in a hypothetical memory-local array.
Activation/bias loads and output stores still use DMA. PIM pays separately for MACs
and local weight reads and contends with DMA for the DRAM port. The comparison uses
a different mapping and placement assumption, not an arbitrary speed multiplier.
See the complete [architecture and timing specification](docs/architecture.md).

## Verified reference benchmarks

Workload: `(16,32) @ (32,24)`; MLP adds bias/Relu then `(16,24) @ (24,12)` and bias.
Weights use seed 11, inputs seed 7. Default clock is 1 GHz, SRAM 64 KiB, tile
`(16,16,16)`, conventional/PIM throughput 64 MACs/cycle, DMA 32 B/cycle, local PIM
reads 128 B/cycle. Setup and epilogue costs are included.

| Case | Cycles | Latency (ns) | Off-chip bytes | PIM-local bytes | Peak SRAM (B) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MatMul conventional | 560 | 560 | 8,704 | 0 | 3,072 |
| MatMul PIM | 456 | 456 | 5,632 | 3,072 | 2,048 |
| MatMul tile `(8,8,8)` | 1,152 | 1,152 | 13,824 | 0 | 768 |
| MLP unoptimized | 1,169 | 1,169 | 19,984 | 0 | 4,608 |
| MLP optimized | 945 | 945 | 13,840 | 0 | 4,096 |
| MLP optimized PIM | 798 | 798 | 9,616 | 4,224 | 3,072 |

These are **simulated accelerator estimates**, not host inference measurements or
real-chip predictions. Off-chip bytes count DMA reads/writes; DRAM reads additionally
include PIM-local reads, reported separately to prevent double counting. The smaller
tile saves SRAM but is slower and moves more bytes. PIM advantages depend on the
configured rates and placement assumptions; they are not universal.

[Full results and utilization](benchmarks/results.json),
[reference environment](benchmarks/environment.json), and
[reproduction details](benchmarks/README.md) are checked in. To refresh figures and
tables from real runs: `python scripts/refresh_artifacts.py` after a dev install.

![Actual optimized MLP native execution timeline](docs/figures/timeline.svg)

## Development and tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check .
clang-format --dry-run --Werror cpp/include/*.hpp cpp/src/*.cpp cpp/tests/*.cpp
cmake -S . -B build/native -DNPUFORGE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
python -m build
```

Reinstall the editable package after C++ changes; Python source changes are immediate.
Tests cover ONNX/IR rejection and invariants, fusion branch safety, dead computation,
broadcasting, tile arithmetic and coverage, memory pressure, deterministic random
graphs, native contention, report accuracy/escaping, CLI failures and complete demos.
Numerical tolerances are **rtol `1e-4`, atol `1e-5`**. Native hand calculations cover
single compute/transfer, dependency serialization, independent overlap, resource
contention, PIM/DRAM sharing, allocation lifetime, deadlock and repeatability.

GitHub Actions runs the build, formatting, Python/native tests, demo and packaging
on macOS and Linux. See [validation evidence](docs/validation.md) for what was actually
executed locally. The MIT-licensed source is published at
[guhanp09/npuforge](https://github.com/guhanp09/npuforge). Hosted build results and
downloadable reports are available in [GitHub Actions](https://github.com/guhanp09/npuforge/actions).

The tested dependency snapshot is available in `requirements-tested.txt` for Python
3.12. To repeat Linux validation locally with Docker (source mounted read-only):

```bash
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD,dst=/source,readonly" \
  python:3.12-slim-bookworm sh /source/scripts/validate_linux.sh
```

## Scope

NPUForge supports a small set of static FP32 operations. Its C++ engine models
execution timing; NumPy executes tensor values. It is not a
silicon-validated or general cycle-accurate simulator. It does not model physical
banks, caches, detailed pipelines, energy, quantization, dynamic shapes, batched
MatMul, convolution or arbitrary ONNX graphs. DRAM capacity is assumed sufficient.
The hardware model is hypothetical and does not represent a commercial chip.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

[MIT](LICENSE).
