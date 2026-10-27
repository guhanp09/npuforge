from copy import deepcopy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from npuforge.compiler import eliminate_dead, fuse, optimize
from npuforge.functional import execute, make_inputs, verify
from npuforge.importer import import_onnx
from npuforge.ir import Graph, Operation, Tensor
from npuforge.models import generate_model


@pytest.mark.parametrize("kind", ["matmul", "mlp"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 7, 5), (17, 33, 19)])
def test_real_import_and_differential(tmp_path, kind, shape):
    path = generate_model(tmp_path / "model.onnx", kind, *shape)
    graph = import_onnx(path)
    assert graph.inputs == ["x"]
    for enabled in (False, True):
        optimized, changes = optimize(graph, enabled)
        assert verify(path, optimized)["passed"]
        assert changes["final_operations"] == len(optimized.operations)
    assert Graph.from_json(graph.to_json()).to_json() == graph.to_json()
    assert str(graph).startswith("graph demo_")


@pytest.mark.parametrize(
    "a,b", [((2, 3), (3,)), ((2, 1, 3), (1, 4, 1)), ((), ()), ((2, 3), ()), ((1,), (3, 2, 1))]
)
def test_add_broadcast(model_factory, a, b):
    out = np.broadcast_shapes(a, b)
    path = model_factory(
        [helper.make_node("Add", ["a", "b"], ["out"])], {"a": a, "b": b}, {"out": out}
    )
    assert verify(path, import_onnx(path))["passed"]


def test_relu(model_factory):
    path = model_factory([helper.make_node("Relu", ["x"], ["out"])], {"x": (3, 2)}, {"out": (3, 2)})
    assert verify(path, import_onnx(path))["passed"]


@pytest.mark.parametrize(
    "op,inputs,output",
    [
        ("Sigmoid", {"a": (2, 3)}, (2, 3)),
        ("MatMul", {"a": (2, 3), "b": (4, 2)}, (2, 2)),
        ("MatMul", {"a": (3,), "b": (3, 2)}, (2,)),
        ("Add", {"a": (2, 3), "b": (4,)}, (2, 3)),
        ("Relu", {"a": ("batch", 3)}, ("batch", 3)),
        ("Relu", {"a": (0, 3)}, (0, 3)),
    ],
)
def test_reject_invalid_or_unsupported(model_factory, op, inputs, output):
    path = model_factory([helper.make_node(op, list(inputs), ["out"])], inputs, {"out": output})
    with pytest.raises(ValueError):
        import_onnx(path)


def test_missing_tensor_and_dtype(model_factory):
    path = model_factory(
        [helper.make_node("Add", ["a", "missing"], ["out"])], {"a": (2,)}, {"out": (2,)}
    )
    with pytest.raises(ValueError, match="invalid ONNX"):
        import_onnx(path)
    model = onnx.load(str(path))
    model.graph.node[0].op_type = "Relu"
    del model.graph.node[0].input[1:]
    for v in [*model.graph.input, *model.graph.output]:
        v.type.tensor_type.elem_type = TensorProto.INT64
    onnx.save(model, str(path))
    with pytest.raises(ValueError, match="FP32"):
        import_onnx(path)


def test_dead_computation(model_factory):
    path = model_factory(
        [
            helper.make_node("Relu", ["x"], ["dead"]),
            helper.make_node("Add", ["x", "x"], ["y"]),
            helper.make_node("Relu", ["dead"], ["also_dead"]),
        ],
        {"x": (2, 3)},
        {"y": (2, 3)},
    )
    graph = import_onnx(path)
    result = eliminate_dead(graph)
    assert len(result.operations) == 1
    assert "dead" not in result.tensors
    assert verify(path, result)["passed"]


def test_fusion_and_observable_intermediates(tmp_path):
    path = generate_model(tmp_path / "mlp.onnx")
    graph = import_onnx(path)
    fused = fuse(graph)
    assert [o.op for o in fused.ordered()] == ["FusedMatMulAddRelu", "MatMul", "Add"]
    assert verify(path, fused)["passed"]
    for output in ("mm1", "bias1"):
        observed = deepcopy(graph)
        observed.outputs.append(output)
        assert not any(o.op.startswith("Fused") for o in fuse(observed).operations)
    branched = deepcopy(graph)
    branched.tensors["branch"] = Tensor("branch", (16, 24))
    branched.operations.append(Operation("branch", "Relu", ["mm1"], "branch"))
    branched.outputs.append("branch")
    assert not any(o.op.startswith("Fused") for o in fuse(branched).operations)


def test_add_self_does_not_fuse(model_factory):
    path = model_factory(
        [
            helper.make_node("MatMul", ["a", "b"], ["mm"]),
            helper.make_node("Add", ["mm", "mm"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["out"]),
        ],
        {"a": (2, 3), "b": (3, 4)},
        {"out": (2, 4)},
    )
    graph = import_onnx(path)
    assert len(fuse(graph).operations) == 3


def test_ir_invariants_and_topological_order():
    tensors = {k: Tensor(k, (2, 2)) for k in ("x", "a", "b")}
    first = Operation("first", "Relu", ["x"], "a")
    second = Operation("second", "Relu", ["a"], "b")
    graph = Graph(tensors, [second, first], ["x"], ["b"])
    graph.validate()
    assert graph.ordered() == [first, second]
    assert tensors["x"].nbytes == 16
    assert graph.consumers()["a"] == [second]
    assert graph.producers()["b"] == second
    for mutation in ("cycle", "duplicate", "missing", "shape", "output", "producer", "dtype"):
        bad = deepcopy(graph)
        if mutation == "cycle":
            bad.operations[1].inputs = ["b"]
        elif mutation == "duplicate":
            bad.operations[1].id = "second"
        elif mutation == "missing":
            bad.operations[0].inputs = ["unknown"]
        elif mutation == "shape":
            bad.tensors["b"].shape = (4,)
        elif mutation == "output":
            bad.outputs = ["unknown"]
        elif mutation == "producer":
            bad.operations[1].output = "x"
        else:
            bad.tensors["x"].dtype = "int8"
        with pytest.raises(ValueError):
            bad.validate()


def test_inputs_validated(tmp_path):
    graph = import_onnx(generate_model(tmp_path / "a.onnx"))
    with pytest.raises(ValueError, match="exactly"):
        execute(graph, {})
    inputs = make_inputs(graph)
    inputs["x"] = inputs["x"].astype(np.float64)
    with pytest.raises(ValueError, match="FP32"):
        execute(graph, inputs)


def test_opset_and_nonfinite_initializer_rejection(tmp_path):
    path = generate_model(tmp_path / "a.onnx", "matmul")
    model = onnx.load(str(path))
    model.opset_import[0].version = 12
    onnx.save(model, str(path))
    with pytest.raises(ValueError, match="opsets"):
        import_onnx(path)
    graph = import_onnx(generate_model(path, "matmul"))
    graph.tensors["w1"].data[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        graph.validate()
