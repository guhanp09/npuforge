"""Independent FP32 semantics; ONNX Runtime is used solely as a reference."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .ir import Graph, gemm_attributes

RTOL = 1e-4
ATOL = 1e-5


def make_inputs(graph: Graph, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        tid: rng.normal(size=graph.tensors[tid].shape).astype(np.float32) for tid in graph.inputs
    }


def execute(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    graph.validate()
    if set(inputs) != set(graph.inputs):
        raise ValueError("functional inputs must exactly match graph inputs")
    values = {k: t.data for k, t in graph.tensors.items() if t.data is not None}
    for tid, value in inputs.items():
        if value.shape != graph.tensors[tid].shape or value.dtype != np.float32:
            raise ValueError(f"{tid}: expected FP32 {graph.tensors[tid].shape}")
        values[tid] = value
    for op in graph.ordered():
        args = [values[t] for t in op.inputs]
        if op.op == "MatMul":
            out = args[0] @ args[1]
        elif op.op in {"Gemm", "FusedGemmRelu"}:
            attrs = gemm_attributes(op)
            a = args[0].T if attrs["transA"] else args[0]
            b = args[1].T if attrs["transB"] else args[1]
            out = np.float32(attrs["alpha"]) * (a @ b)
            if len(args) == 3:
                out = out + np.float32(attrs["beta"]) * args[2]
            if op.op == "FusedGemmRelu":
                out = np.maximum(out, np.float32(0))
        elif op.op == "Add":
            out = args[0] + args[1]
        elif op.op == "Relu":
            out = np.maximum(args[0], np.float32(0))
        elif op.op == "FusedMatMulAddRelu":
            out = np.maximum(args[0] @ args[1] + args[2], np.float32(0))
        else:
            raise ValueError(f"unsupported functional operation {op.op}")
        values[op.output] = out
    return {tid: values[tid] for tid in graph.outputs}


def verify(path: str | Path, graph: Graph, seed: int = 7) -> dict:
    inputs = make_inputs(graph, seed)
    actual = execute(graph, inputs)
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        expected = session.run(graph.outputs, inputs)
    except Exception as exc:
        # ORT's native exception classes do not share a public Python base class.
        raise ValueError(f"ONNX Runtime reference could not load or execute model: {exc}") from exc
    errors = {}
    for tid, reference in zip(graph.outputs, expected, strict=True):
        if not np.all(np.isfinite(actual[tid])) or not np.all(np.isfinite(reference)):
            raise ValueError(f"{tid}: non-finite output; check input range and weights")
        np.testing.assert_allclose(actual[tid], reference, rtol=RTOL, atol=ATOL, equal_nan=False)
        errors[tid] = float(np.max(np.abs(actual[tid] - reference)))
    return {"passed": True, "seed": seed, "rtol": RTOL, "atol": ATOL, "max_absolute_error": errors}
