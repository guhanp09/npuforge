from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


@pytest.fixture
def model_factory(tmp_path):
    def create(nodes, inputs, outputs, weights=None, name="case") -> Path:
        values = lambda items: [  # noqa: E731
            helper.make_tensor_value_info(key, TensorProto.FLOAT, shape)
            for key, shape in items.items()
        ]
        initializers = [
            numpy_helper.from_array(np.asarray(data, dtype=np.float32), key)
            for key, data in (weights or {}).items()
        ]
        graph = helper.make_graph(nodes, name, values(inputs), values(outputs), initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        path = tmp_path / f"{name}.onnx"
        onnx.save(model, str(path))
        return path

    return create


@pytest.fixture
def gemm_factory(model_factory):
    def create(
        *,
        attrs=None,
        bias_shape=None,
        explicit_empty=False,
        constants=False,
        m=3,
        k=7,
        n=5,
        relu=False,
    ):
        attrs = attrs or {}
        inputs = {
            "a": (k, m) if attrs.get("transA", 0) else (m, k),
            "b": (n, k) if attrs.get("transB", 0) else (k, n),
        }
        names = ["a", "b"]
        if bias_shape is not None:
            inputs["c"] = bias_shape
            names.append("c")
        elif explicit_empty:
            names.append("")
        weights = {}
        if constants:
            rng = np.random.default_rng(41)
            for tid in ("b", "c"):
                if tid in inputs:
                    weights[tid] = rng.normal(size=inputs.pop(tid)).astype(np.float32)
        nodes = [helper.make_node("Gemm", names, ["y"], **attrs)]
        if relu:
            nodes.append(helper.make_node("Relu", ["y"], ["out"]))
        return model_factory(nodes, inputs, {"out" if relu else "y": (m, n)}, weights, name="gemm")

    return create
