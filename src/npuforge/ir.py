"""Validated, inspectable IR independent of ONNX protobufs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tensor:
    id: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    data: np.ndarray | None = field(default=None, repr=False)

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * 4


@dataclass
class Operation:
    id: str
    op: str
    inputs: list[str]
    output: str
    attrs: dict = field(default_factory=dict)


def gemm_attributes(op: Operation) -> dict:
    """Normalize the ONNX Gemm defaults without accepting unmodeled attributes."""
    defaults = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
    metadata = {"fused_from"} if op.op == "FusedGemmRelu" else set()
    if op.attrs.keys() - defaults.keys() - metadata:
        raise ValueError(f"{op.id}: unsupported Gemm attributes")
    attrs = defaults | {k: v for k, v in op.attrs.items() if k not in metadata}
    for key in ("alpha", "beta"):
        value = attrs[key]
        if (
            type(value) not in (int, float)
            or abs(value) > float(np.finfo(np.float32).max)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{op.id}: Gemm {key} requires a finite FP32 scalar")
        attrs[key] = float(np.float32(value))
    for key in ("transA", "transB"):
        if type(attrs[key]) is not int:
            raise ValueError(f"{op.id}: Gemm {key} requires an integer transpose flag")
    return attrs


def result_shape(op: Operation, tensors: dict[str, Tensor]) -> tuple[int, ...]:
    shapes = [tensors[t].shape for t in op.inputs]
    if op.op in {"Gemm", "FusedGemmRelu"}:
        attrs = gemm_attributes(op)
        if len(shapes) not in (2, 3) or any(len(s) != 2 for s in shapes[:2]):
            raise ValueError(f"{op.id}: Gemm requires two rank-2 matrices and optional bias C")
        a = shapes[0][::-1] if attrs["transA"] else shapes[0]
        b = shapes[1][::-1] if attrs["transB"] else shapes[1]
        if a[1] != b[0]:
            raise ValueError(f"{op.id}: incompatible Gemm reduction dimensions {a}, {b}")
        out = (a[0], b[1])
        if len(shapes) == 3 and np.broadcast_shapes(out, shapes[2]) != out:
            raise ValueError(f"{op.id}: Gemm bias C must broadcast into {out}")
        return out
    if op.op in {"MatMul", "FusedMatMulAddRelu"}:
        arity = 2 if op.op == "MatMul" else 3
        if len(shapes) != arity or len(shapes[0]) != 2 or len(shapes[1]) != 2:
            raise ValueError(f"{op.id}: MatMul requires two rank-2 matrices")
        if shapes[0][1] != shapes[1][0]:
            raise ValueError(f"{op.id}: incompatible MatMul dimensions {shapes[:2]}")
        out = (shapes[0][0], shapes[1][1])
        if arity == 3 and np.broadcast_shapes(out, shapes[2]) != out:
            raise ValueError(f"{op.id}: fused bias must broadcast into MatMul output")
        return out
    if op.op == "Add" and len(shapes) == 2:
        return np.broadcast_shapes(*shapes)
    if op.op == "Relu" and len(shapes) == 1:
        return shapes[0]
    raise ValueError(f"{op.id}: unsupported operation or arity: {op.op}")


@dataclass
class Graph:
    tensors: dict[str, Tensor]
    operations: list[Operation]
    inputs: list[str]
    outputs: list[str]
    name: str = "graph"

    def producers(self) -> dict[str, Operation]:
        return {op.output: op for op in self.operations}

    def consumers(self) -> dict[str, list[Operation]]:
        result: dict[str, list[Operation]] = {t: [] for t in self.tensors}
        for op in self.operations:
            for tid in set(op.inputs):
                result[tid].append(op)
        return result

    def ordered(self) -> list[Operation]:
        available = set(self.inputs) | {t.id for t in self.tensors.values() if t.data is not None}
        pending = list(self.operations)
        ordered = []
        while pending:
            ready = [op for op in pending if set(op.inputs) <= available]
            if not ready:
                raise ValueError("graph contains a cycle or missing tensor producer")
            for op in ready:
                ordered.append(op)
                available.add(op.output)
                pending.remove(op)
        return ordered

    def validate(self) -> None:
        if not self.outputs:
            raise ValueError("graph requires at least one output")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate graph inputs or outputs")
        if len({op.id for op in self.operations}) != len(self.operations):
            raise ValueError("duplicate operation ID")
        available = set(self.inputs) | {t.id for t in self.tensors.values() if t.data is not None}
        produced = set()
        for key, tensor in self.tensors.items():
            if not key or key != tensor.id or tensor.dtype != "float32":
                raise ValueError("tensor IDs must match keys; only FP32 is supported")
            if any(type(d) is not int or d <= 0 for d in tensor.shape):
                raise ValueError(f"{key}: requires positive static dimensions")
            if tensor.data is not None and (
                tensor.data.shape != tensor.shape or tensor.data.dtype != np.float32
            ):
                raise ValueError(f"{key}: initializer shape or dtype mismatch")
            if tensor.data is not None and not np.all(np.isfinite(tensor.data)):
                raise ValueError(f"{key}: non-finite initializer is unsupported")
            if tensor.data is not None and key in self.inputs:
                raise ValueError(f"{key}: graph inputs and initializers must be disjoint")
        for tid in self.inputs + self.outputs:
            if tid not in self.tensors:
                raise ValueError(f"missing tensor: {tid}")
        for op in self.operations:
            if not op.id or op.output not in self.tensors:
                raise ValueError("operation requires an ID and declared output")
            if op.output in available or op.output in produced:
                raise ValueError(f"multiple producers for {op.output}")
            produced.add(op.output)
            if any(t not in self.tensors for t in op.inputs):
                raise ValueError(f"{op.id}: missing input tensor")
            if result_shape(op, self.tensors) != self.tensors[op.output].shape:
                raise ValueError(f"{op.id}: output shape mismatch")
        self.ordered()
        if not set(self.outputs) <= available | produced:
            raise ValueError("graph output has no producer")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema": "npuforge.ir.v1",
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tensors": [
                {
                    "id": t.id,
                    "shape": list(t.shape),
                    "dtype": t.dtype,
                    "data": None if t.data is None else t.data.tolist(),
                }
                for t in sorted(self.tensors.values(), key=lambda t: t.id)
            ],
            "operations": [
                {"id": o.id, "op": o.op, "inputs": o.inputs, "output": o.output, "attrs": o.attrs}
                for o in self.ordered()
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Graph:
        d = json.loads(text)
        if d.get("schema") != "npuforge.ir.v1":
            raise ValueError("unsupported IR schema")
        tensors = {}
        for t in d["tensors"]:
            if t["id"] in tensors:
                raise ValueError("duplicate serialized tensor")
            data = None if t["data"] is None else np.asarray(t["data"], dtype=np.float32)
            tensors[t["id"]] = Tensor(t["id"], tuple(t["shape"]), t["dtype"], data)
        graph = cls(
            tensors, [Operation(**o) for o in d["operations"]], d["inputs"], d["outputs"], d["name"]
        )
        graph.validate()
        return graph

    def __str__(self) -> str:
        lines = [f"graph {self.name} ({', '.join(self.inputs)}) -> ({', '.join(self.outputs)})"]
        for op in self.ordered():
            attributes = f" {json.dumps(op.attrs, sort_keys=True)}" if op.attrs else ""
            lines.append(
                f"  {op.output}{self.tensors[op.output].shape} = "
                f"{op.op}({', '.join(op.inputs)}){attributes}"
            )
        return "\n".join(lines)
