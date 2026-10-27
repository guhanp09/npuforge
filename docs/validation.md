# Validation

The current suite has **182 Python tests** and a native C++ test executable.
The [GitHub Actions workflow](https://github.com/guhanp09/npuforge/actions/workflows/ci.yml)
builds the package and runs formatting checks, tests, examples and packaging on
Ubuntu and macOS. Hosted checks passed on both platforms on 27 October 2026;
the workflow page contains the current run status and downloadable reports.

## Tested configurations

| Platform | Python | Checks |
| :--- | :--- | :--- |
| macOS arm64 | 3.12 | 182 tests, native CTest, Ruff, clang-format, examples, source/wheel builds |
| macOS arm64 | 3.11 | Package installation, 182 tests and example comparisons |
| Linux x86_64, Debian Bookworm container | 3.12 | Clean installation, 182 tests, native CTest, formatting, examples and packaging |

A production wheel was also installed and exercised outside the source directory.
Dependency checks passed. The native tests passed in a Debug build with
AddressSanitizer and UndefinedBehaviorSanitizer.

## Coverage

- ONNX import, unsupported operators, malformed files, static shapes and FP32 types.
- IR producer/consumer invariants, ordering, cycles and deterministic JSON round trips.
- Dead computation elimination and fusion safety for shared or public intermediates.
- MatMul and Gemm tile coverage, K accumulation, transpose gathers, broadcast bias
  and SRAM capacity. Instruction replay reconstructs outputs from emitted tile metadata.
- Gemm defaults, finite coefficients, optional C, transpose flags and final-tile activation.
- Native dependency scheduling, compute/DMA overlap, resource contention, allocation
  lifetimes, working-set bounds, invalid frees and allocation deadlock.
- CLI errors, report metrics, HTML escaping, SVG structure and repeatable benchmark output.

NumPy outputs are compared with ONNX Runtime CPU using `rtol=1e-4, atol=1e-5`.
Tests use fixed random seeds. Non-finite initializers and outputs are rejected.

The C++ tests use hand-calculated cases for single transfers and computations,
sequential dependencies, independent resource overlap, compute contention and the
shared PIM/DMA memory port. Python integration tests check Gemm timing with a
`(2,3) @ (3,4)` workload, alpha=2, beta=-1 and vector bias:

| Configuration | Conventional cycles | PIM cycles |
| :--- | ---: | ---: |
| Gemm | 41 | 38 |
| Gemm followed by Relu | 59 | 56 |
| Fused Gemm/Relu | 43 | 40 |

A cancellation test catches activation applied before the final reduction. Other
regressions cover colliding fusion IDs, published intermediate values and multiple
consumers of the activated output.

## Reproducing Linux validation

From the repository root, with Docker running:

```bash
docker run --rm --platform linux/amd64 --cpus=2 --memory=2g \
  --mount "type=bind,src=$PWD,dst=/source,readonly" \
  python:3.12-slim-bookworm sh /source/scripts/validate_linux.sh
```

The script copies the source into the container, installs the package, runs the
checks and compares all ten native metric dictionaries with the reference JSON in
`benchmarks/`. The macOS and Linux results matched exactly for those workloads.
FP32 verification error values can differ slightly across BLAS implementations.

Source archives are checked for required source/test files and exclude bytecode,
virtual environments, generated reports and native build output. The timing model's
assumptions and limitations are described in [architecture.md](architecture.md).
