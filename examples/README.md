# Examples

After installation, `npuforge demo` generates and runs the MatMul and MLP ONNX models.
The standalone generator also creates a Gemm dense layer with transposed weights,
alpha 0.75, beta 0.5 and a vector bias, plus a Gemm/Relu model for fusion comparison.
To inspect compilation manually:

```bash
python examples/generate_models.py --output output/models
npuforge inspect output/models/demo_mlp.onnx
npuforge compile output/models/demo_mlp.onnx --no-opt --output output/baseline
npuforge compile output/models/demo_mlp.onnx --compare --output output/optimized
npuforge compile output/models/demo_matmul.onnx --config configs/pim.json --output output/pim
npuforge compile output/models/demo_gemm.onnx --compare --output output/gemm
python examples/compare_gemm_fusion.py
```

Generated models and artifacts stay in ignored `output/`. No downloaded network
weights, specialized accelerator hardware, external account or service is required.

The fusion comparison produces one report containing conventional/PIM, each with
optimization disabled and enabled. Every case is compiled, simulated natively and
verified against ONNX Runtime. Override its artifact directory with `--output`.
