"""Small real ONNX fixtures with deterministic FP32 weights."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def generate_model(
    path: str | Path, kind: str = "mlp", m: int = 16, k: int = 32, n: int = 24, seed: int = 11
) -> Path:
    if kind not in {"matmul", "mlp", "gemm", "gemm_relu"} or min(m, k, n) <= 0:
        raise ValueError("model must be matmul, mlp, gemm or gemm_relu with positive dimensions")
    rng = np.random.default_rng(seed)
    weights = [numpy_helper.from_array((rng.normal(size=(k, n)) * 0.2).astype(np.float32), "w1")]
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["mm1"], name="linear1")]
    out, out_shape = "mm1", [m, n]
    if kind in {"gemm", "gemm_relu"}:
        # Common dense-layer storage: weights are output-by-input, consumed with transB.
        weights[0] = numpy_helper.from_array(numpy_helper.to_array(weights[0]).T.copy(), "w1")
        weights.append(
            numpy_helper.from_array((rng.normal(size=(n,)) * 0.2).astype(np.float32), "b1")
        )
        nodes = [
            helper.make_node(
                "Gemm", ["x", "w1", "b1"], ["y"], name="dense", transB=1, alpha=0.75, beta=0.5
            )
        ]
        out = "y"
        if kind == "gemm_relu":
            nodes.append(helper.make_node("Relu", ["y"], ["activated"]))
            out = "activated"
    if kind == "mlp":
        p = max(2, n // 2)
        for name, shape in [("b1", (n,)), ("w2", (n, p)), ("b2", (p,))]:
            weights.append(
                numpy_helper.from_array((rng.normal(size=shape) * 0.2).astype(np.float32), name)
            )
        nodes.extend(
            [
                helper.make_node("Add", ["mm1", "b1"], ["bias1"]),
                helper.make_node("Relu", ["bias1"], ["hidden"]),
                helper.make_node("MatMul", ["hidden", "w2"], ["mm2"]),
                helper.make_node("Add", ["mm2", "b2"], ["y"]),
            ]
        )
        out, out_shape = "y", [m, p]
    graph = helper.make_graph(
        nodes,
        f"demo_{kind}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, out_shape)],
        weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # IR 9 is deliberately portable across the supported ONNX Runtime releases.
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
