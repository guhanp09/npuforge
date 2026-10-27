# Compiler and abstract hardware model

NPUForge compiles actual ONNX into its own typed tensor graph, rewrites that graph,
and lowers it to an explicit instruction DAG. The **C++ engine is a performance
simulator**, not a tensor arithmetic engine. Independent NumPy execution implements
FP32 graph semantics; ONNX Runtime supplies the differential reference. Tests also
reconstruct numerical outputs from lowered tile instructions to check mapping.

## Supported input contract

- Default-domain ONNX opsets 13–18. Models generated here use opset 18 and IR 9.
- Finite FP32 initializers; positive, fully static dimensions. Scalars are supported.
- `MatMul`: exactly `(M, K) @ (K, N) -> (M, N)`. No vectors, batches or transposes.
- `Gemm`: rank-2 matrices with `Y = alpha * op(A) @ op(B) + beta * C`, where
  `op` transposes when the corresponding integer `transA`/`transB` flag is nonzero.
  Defaults are alpha=beta=1 and transA=transB=0. C is optional (omitted or an empty
  trailing input) and must broadcast into `(M,N)` without expanding the result.
  Scalar, row, column and full-matrix biases are supported. Alpha/beta must be finite
  FP32 scalars. B and C can be constants or runtime tensors. This follows the
  [ONNX Gemm-13 specification](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-13).
- `Add`: full NumPy/modern ONNX multidirectional broadcasting, any static rank,
  including scalar and both-sided expansion.
- `Relu`: any supported static shape, `maximum(x, 0)`.
- No custom domains, local functions, sparse weights, dynamic shapes or attributes
  other than the four standard Gemm attributes.
  Unsupported operations fail import, including dead unsupported operations.
- Initializers are immutable constants; initializer names also listed in ONNX inputs
  are treated as constants, excluded from the runtime feed contract.

The ONNX checker and shape inference run before the independent IR validator. The
IR checks sources, shapes, arity, FP32 types, unique IDs/producers, acyclicity and
public outputs. Producer/consumer maps and stable topological order are derived
from tensor identity. Sorted JSON includes weights, metadata and fused provenance,
so exporting and importing the IR preserves its semantics.

## Transformations

Dead computation elimination starts from **all public outputs**, walks producer
edges backwards, and removes unreachable nodes and constants. It retains public
runtime inputs, even if unused, preserving the reference session's input contract.

Fusion replaces `MatMul -> Add -> Relu` with `FusedMatMulAddRelu`. Either operand
of Add can be the matrix result. Bias must broadcast **into** the matrix output
without expanding that output. Both intermediates must have exactly one consumer
and must not be public outputs. `Add(x, x)` is excluded. The fused IR carries all
three inputs and provenance; the final K tile executes the epilogue exactly once.

Use `compile --no-opt` to disable both passes. The MLP has five original operations
and three optimized operations. The last MatMul/Add remains separate because there
is no trailing Relu. The additional Gemm/Relu fusion pattern is described below.

## Placement and tiling

All graph inputs, constants and inter-operation outputs initially/live in DRAM.
There is no persistent SRAM cache. Every operation writes its output to DRAM;
its consumers depend on all producing stores. A single reusable tile workspace
serializes compiled tiles and operations. This is conservative, straightforward
mapping; the simulator itself supports independent instruction overlap.

For a conventional `(a, c) @ (c, b)` tile, reserved bytes are:

```
4 * (a*c + c*b + a*b + (a*b if fused else 0))
```

These terms are activation staging, weight staging, FP32 accumulator, and optional
expanded bias staging. PIM removes only the `c*b` **SRAM** term; local weight reads
remain accounted for. Bias staging reserves a full output tile even if the bias is
smaller, making the bound conservative. DMA reads count only unique bias values.

Requested tile sizes are upper bounds. The tiler clamps to matrix dimensions and
repeatedly halves the axis giving the largest memory reduction until the complete
workspace fits. It rejects configurations too small for a 1×1×1 tile. Boundary tiles
use actual extents. K chunks reuse staging while preserving the accumulator; they
depend on the preceding chunk. No DRAM spill is needed inside a reduction. The
accumulator initialization is included in compute setup; ALLOC does not model a
separate memset. MatMul counts one multiply-accumulate as **one MAC**, not two FLOPs.

Elementwise operations flatten the output into chunks of at most 8,192 elements.
For `q` inputs and `e` output elements, SRAM is `4*(q+1)*e`. Broadcasting expands
locally; transfer counts use unique source positions within each chunk. No reuse
between chunks or across repeated operands is assumed. Gather/scatter and broadcast
address generation carry no extra penalty beyond DMA setup and transferred bytes.
DRAM capacity is assumed sufficient; only scratchpad capacity is constrained.

## Instruction contract

IDs are dense integers in storage order. Dependencies may point forwards or
backwards but must form a DAG. Metadata retains node/tensor IDs and tile ranges.
All compiler-generated data instructions reference an active workspace allocation.

| Instruction | Semantics | Exclusive resources |
| :--- | :--- | :--- |
| `ALLOC` | Reserve bytes under a buffer ID until FREE; wait if capacity is unavailable | SRAM reservation |
| `FREE` | Release that allocation after all its uses | SRAM reservation |
| `DMA_LOAD` | Gather the specified source tile from DRAM into staging | DMA + DRAM port |
| `DMA_STORE` | Scatter output tile from SRAM to DRAM | DMA + DRAM port |
| `MATMUL` | Initialize/accumulate the specified K slice | Compute |
| `ADD` | Broadcast and add a flattened output chunk | Compute |
| `RELU` | Apply maximum with zero to a chunk | Compute |
| `FUSED_MATMUL_ADD_RELU` | Final K chunk, then bias and Relu, without intermediate DRAM stores | Compute |
| `PIM_MATMUL` | Memory-local B tile times SRAM A tile; optional fused final epilogue | PIM + DRAM port |

Standalone resource timing microbenchmarks may omit SRAM (`buffer=-1`) and model
register-only work. The compiler never does so. Native checks reject invalid
IDs/dependencies, cycles, unsupported opcodes, capacity overflow, unallocated uses,
oversized transfer/working sets, free during active use, double allocation,
unreleased buffers and SRAM allocation deadlock. Counts/cycle addition are checked
for overflow. The compiler, rather than the scheduler, defines tile address and
accumulation semantics; differential lowering tests check that boundary.

## Deterministic event scheduling

One non-preemptive compute engine, one DMA engine, one PIM engine, and one shared
DRAM port are available. Conventional compute can overlap DMA. PIM and DMA cannot
overlap because they share the DRAM port. Conventional compute and PIM can overlap
in an independently constructed legal workload. Hardware rates never vary at runtime.

At each event cycle, complete all finishing instructions and make their consumers
eligible. Scan eligible instructions in stable ascending ID order, issue those
whose resources and allocation are available, and repeat to resolve zero-cycle
ALLOC/FREE chains. Resource reservations last until completion. If work remains,
advance to the earliest completion; if nothing can run, diagnose allocation deadlock.
A ready ALLOC can wait for a later eligible FREE rather than overcommit SRAM.

## Timing formulas

All nonzero instructions take at least one cycle; ALLOC and FREE take zero.
`ceildiv(x, r)` is integer ceiling division. Default hardware:

| Parameter | Default |
| :--- | ---: |
| Clock | 1 GHz |
| SRAM | 65,536 B |
| Conventional MACs/cycle | 64 |
| Vector elements/cycle | 16 |
| DMA bytes/cycle | 32 |
| DMA setup | 8 cycles |
| Compute/PIM setup | 4 cycles |
| PIM MACs/cycle | 64 |
| PIM local weight bytes/cycle | 128 |
| Requested MatMul tile (M,N,K) | (16,16,16) |

```
DMA = dma_setup + ceildiv(bytes, dma_bytes_per_cycle)
MATMUL = compute_setup + ceildiv(MACs, macs_per_cycle)
ADD or RELU = compute_setup + ceildiv(elements, elements_per_cycle)
FUSED = MATMUL + ceildiv(2*output_tile_elements, elements_per_cycle)
PIM = compute_setup + ceildiv(MACs, pim_macs_per_cycle)
      + ceildiv(local_weight_bytes, pim_bytes_per_cycle)
      + ceildiv(epilogue_elements, elements_per_cycle)
latency_ns = makespan_cycles * 1e9 / clock_hz
```

PIM's local weight read and arithmetic costs are **added**, a conservative
non-overlapped abstraction inside that instruction. Activation/bias DMA and output
stores remain explicit. PIM assumes local B values already reside in its DRAM array;
this holds for inputs/constants initially in DRAM and outputs from earlier stores.
It does not assume zero-cost reads, persistent on-chip weights, or an arbitrary
speed multiplier. Changing bandwidths, rates, setup or tiling can reverse a result.
The parameters describe a hypothetical architecture and are not calibrated to
commercial hardware.

## Gemm mapping

Gemm remains an explicit operation in unoptimized IR, with its original attributes
preserved in JSON and readable graph printing. Dead computation elimination handles
Gemm. The MatMul/Add/Relu pattern does not match Gemm; a separate
Gemm/Relu pattern with the same observability safeguards.

The lowerer tiles the effective matrices after transpose flags have been applied.
DMA metadata includes source row/column ranges and whether to transpose the gathered
tile. PIM instructions carry equivalent source ranges for the memory-local B tile.
The abstract hardware supports these address patterns with the same bandwidth/setup
formula as other gathers: layout conversion, stride penalties and bank conflicts
are not separately modeled. No full transposed tensor is allocated or transferred.

K slices accumulate before scaling. On the final K slice of each output tile,
multiply the accumulator by alpha (unless alpha=1); if C exists, gather the unique
bias values, multiply by beta (unless beta=1), and add to the accumulator. C gets an
extra full output tile's SRAM staging, just like a fused bias. No C buffer, transfer
or beta arithmetic is emitted when C is absent. Zero coefficients still execute
their declared arithmetic and transfers; no extra constant-folding pass is assumed.

Conventional execution uses `MATMUL` and PIM uses `PIM_MATMUL`, with explicit Gemm
attributes and a final-epilogue flag in the instruction metadata. The existing
native `elements` counter charges the scalar operations. Additional final-tile work:

```
scalar_ops_per_element = (alpha != 1)
                        + (1 + (beta != 1) if C exists else 0)
                        + (1 if Relu is fused else 0)
epilogue_cycles = ceildiv(output_tile_elements * scalar_ops_per_element,
                         elements_per_cycle)
```

This is added once per output tile to the final matrix instruction, including in
PIM mode. It does not alter dependencies or make memory-local reads free. The
generated Gemm example is `(16,32)` times stored `(24,32)` with transB=1, a `(24,)`
bias, alpha=0.75 and beta=0.5. Default simulated results: conventional 651 cycles,
8,800 off-chip bytes and 4,096 B peak SRAM; PIM 547 cycles, 5,728 off-chip bytes,
3,072 local weight-read bytes and 3,072 B peak SRAM.

## Gemm/Relu fusion

Replace `Gemm -> Relu` with `FusedGemmRelu` only when the Gemm result has exactly one
consumer and is not a public output. The Relu output's name, shape and consumers are
preserved, even if it has multiple consumers. Original Gemm attributes and the two
source operation IDs are retained. Fused operation IDs cannot collide with existing
IDs. Applying the pass twice is idempotent. `--no-opt` preserves both original nodes.

The fused functional path computes the entire matrix reduction, alpha scaling,
optional beta-scaled C addition, and finally Relu. The backend keeps the accumulator
live through that complete epilogue. Only the final K tile carries `gemm_relu=true`,
and one extra vector element operation per output value is included in `elements`.
No additional SRAM buffer is needed for the in-place activation. The Gemm output
store, standalone Relu load and standalone Relu compute/store are replaced by one
final activated output store, removing one intermediate store/load round trip.

The existing `MATMUL`/`PIM_MATMUL` native instructions execute the aggregate matrix
and vector work; this adds no free activation or arbitrary timing multiplier.
The example generated by `python examples/compare_gemm_fusion.py` reports all four
configurations in one HTML document. At the default rates, conventional execution
changes from 791 to 675 cycles and off-chip traffic from 11,872 to 8,800 B; PIM changes
from 687 to 571 cycles and 8,800 to 5,728 B. Peak SRAM and local weight reads remain
unchanged for this workload. These are model-specific results, not a universal
fusion speedup guarantee. The original six benchmark configurations are unchanged.

## Metrics and integrity

- Total cycles are the last completion timestamp, **not** summed instruction time.
- DMA off-chip movement = DMA load bytes + DMA store bytes.
- DRAM read bytes = DMA load bytes + PIM-local weight read bytes.
- DRAM write bytes = DMA store bytes. PIM-local bytes are also reported separately;
  do not add them to total DRAM reads again.
- Peak SRAM is the maximum sum of live ALLOC reservations, not tensor model size.
- Each utilization is busy resource cycles / makespan. Compute and PIM are separate.
  DRAM busy includes DMA and PIM. Utilizations across shared resources must not be summed.
- Per-operation start/end shows elapsed span; busy durations can overlap. A timeline
  displays actual issue/completion times. At most 1,000 active instructions are drawn;
  the full trace remains in `simulation.json`.
- Bottleneck is the larger of DRAM-port busy time and combined compute/PIM busy time,
  with an equality label. This is a pressure heuristic, not critical-path analysis;
  serialized mapping and dependencies may also dominate. PIM occupies both its
  arithmetic engine and the DRAM port, so this label is deliberately conservative.

NumPy functional results use `rtol=1e-4, atol=1e-5` against ONNX Runtime CPU. Seeds are
recorded; non-finite outputs fail verification. This tolerance permits ordinary FP32
BLAS/FMA ordering differences. The performance engine models counts and schedules,
not arithmetic rounding, physical banks, caches, pipelines, power or host overhead.
No simulated latency should be presented as measured host inference time, silicon
latency, or universal cycle accuracy.
