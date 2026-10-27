# Changelog

## 0.3.0

- Fuse Gemm followed by Relu when the intermediate result is neither shared nor public.
- Keep activation in the final matrix tile and include its vector work in simulation.
- Add a four-case conventional/PIM comparison with instruction traces and an HTML report.
- Avoid operation ID collisions during fusion.

## 0.2.0

- Support static FP32 Gemm with transpose flags, alpha/beta scaling and optional bias.
- Lower transposed matrix tiles to DMA gathers or memory-local PIM reads.
- Add numerical, tiling and timing tests for Gemm and a dense-layer example.

## 0.1.0

- Import static FP32 MatMul, Add and Relu graphs from ONNX.
- Add tensor IR, dead computation elimination and MatMul/Add/Relu fusion.
- Tile matrix operations to fit SRAM and emit explicit DMA and allocation instructions.
- Schedule instructions in a C++ resource simulator with conventional and PIM mappings.
- Add ONNX Runtime verification, benchmarks, reports and macOS/Linux CI.
