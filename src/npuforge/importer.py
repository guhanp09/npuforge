"""Strict static FP32 ONNX import: rank-2 MatMul/Gemm, Add and Relu."""

from pathlib import Path

import onnx
from google.protobuf.message import DecodeError
from onnx import TensorProto, helper, numpy_helper

from .ir import Graph, Operation, Tensor


def import_onnx(path: str | Path) -> Graph:
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=True)
        opsets = {o.domain: o.version for o in model.opset_import}
        if not 13 <= opsets.get("", opsets.get("ai.onnx", 0)) <= 18:
            raise ValueError("supported default-domain ONNX opsets are 13 through 18")
        if model.functions or model.graph.sparse_initializer:
            raise ValueError("local functions and sparse initializers are not supported")
        for node in model.graph.node:
            if node.domain not in {"", "ai.onnx"} or node.op_type not in {
                "MatMul",
                "Gemm",
                "Add",
                "Relu",
            }:
                raise ValueError(f"unsupported ONNX operation: {node.domain}:{node.op_type}")
            if (node.attribute and node.op_type != "Gemm") or len(node.output) != 1:
                raise ValueError(f"{node.op_type}: attributes and multiple outputs are unsupported")
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        tensors = {}
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
            tt = value.type.tensor_type
            if tt.elem_type != TensorProto.FLOAT or not tt.HasField("shape"):
                raise ValueError(f"{value.name}: requires FP32 and a known static shape")
            shape = []
            for dim in tt.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    raise ValueError(
                        f"{value.name}: dynamic, zero or unresolved shape is unsupported"
                    )
                shape.append(int(dim.dim_value))
            tensors[value.name] = Tensor(value.name, tuple(shape))
        for initializer in model.graph.initializer:
            if initializer.data_type != TensorProto.FLOAT:
                raise ValueError(f"{initializer.name}: only FP32 initializers are supported")
            data = numpy_helper.to_array(initializer).copy()
            if initializer.name in tensors and tensors[initializer.name].shape != data.shape:
                raise ValueError(f"{initializer.name}: conflicting initializer metadata")
            tensors[initializer.name] = Tensor(initializer.name, data.shape, data=data)
        operations = []
        for idx, node in enumerate(model.graph.node):
            inputs = list(node.input)
            # ONNX allows either omitted C or an explicit empty trailing argument.
            if node.op_type == "Gemm" and len(inputs) == 3 and inputs[-1] == "":
                inputs.pop()
            operations.append(
                Operation(
                    f"op{idx}_{node.name or node.op_type}",
                    node.op_type,
                    inputs,
                    node.output[0],
                    {attr.name: helper.get_attribute_value(attr) for attr in node.attribute},
                )
            )
        inputs = [i.name for i in model.graph.input if tensors[i.name].data is None]
        graph = Graph(
            tensors,
            operations,
            inputs,
            [o.name for o in model.graph.output],
            model.graph.name or Path(path).stem,
        )
        graph.validate()
        return graph
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError, DecodeError) as exc:
        raise ValueError(f"invalid ONNX model: {exc}") from exc
