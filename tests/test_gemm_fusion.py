"""Observable-value safety and real tile/traffic effects of Gemm -> Relu fusion."""

from copy import deepcopy

import numpy as np
import pytest
from onnx import helper

from npuforge.backend import Hardware, lower, simulate
from npuforge.benchmark import run_fusion_benchmarks
from npuforge.compiler import fuse, optimize
from npuforge.functional import execute, make_inputs, verify
from npuforge.importer import import_onnx
from npuforge.ir import Graph, Operation, Tensor
from npuforge.models import generate_model
from npuforge.reports import render_report


def replay_fused(program, graph, inputs):
    """Reconstruct one fused output using emitted source addresses and epilogue flags."""
    values = {tid: t.data for tid, t in graph.tensors.items() if t.data is not None} | inputs
    output = np.zeros(graph.tensors[graph.outputs[0]].shape, dtype=np.float32)
    staging, accumulators = {}, {}
    for i in program.instructions:
        if i["op"] == "DMA_LOAD" and "source_ranges" in i:
            source = i["source_ranges"]
            tile = values[i["tensor"]][slice(*source["rows"]), slice(*source["cols"])]
            staging[i["buffer"], i["tensor"]] = tile.T if i["transpose"] else tile
        elif i["op"] in {"MATMUL", "PIM_MATMUL"}:
            a_id, b_id = i["inputs"][:2]
            if i["op"] == "PIM_MATMUL":
                source = i["local_source_ranges"]
                b = values[b_id][slice(*source["rows"]), slice(*source["cols"])]
                b = b.T if i["local_transpose"] else b
            else:
                b = staging[i["buffer"], b_id]
            partial = staging[i["buffer"], a_id] @ b
            acc = accumulators[i["buffer"]] + partial if i["accumulate"] else partial
            if i["gemm_epilogue"]:
                ms, ns = (slice(*i["ranges"][axis]) for axis in ("m", "n"))
                acc = np.float32(i["gemm"]["alpha"]) * acc
                if len(i["inputs"]) == 3:
                    bias = np.broadcast_to(values[i["inputs"][2]], output.shape)[ms, ns]
                    acc += np.float32(i["gemm"]["beta"]) * bias
            if i["gemm_relu"]:
                assert i["gemm_epilogue"]
                acc = np.maximum(acc, np.float32(0))
            else:
                assert not i["gemm_epilogue"]
                assert i["elements"] == 0
            accumulators[i["buffer"]] = acc
        elif i["op"] == "DMA_STORE":
            ms, ns = (slice(*i["ranges"][axis]) for axis in ("m", "n"))
            output[ms, ns] = accumulators[i["buffer"]]
    return output


@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("bias", [None, (), (5,), (3, 1), (3, 5)])
def test_fused_ir_and_tiled_execution(gemm_factory, architecture, bias):
    path = gemm_factory(
        attrs={"transA": 1, "transB": 1, "alpha": -0.5, "beta": 2.0}, bias_shape=bias, relu=True
    )
    original = import_onnx(path)
    before = original.to_json()
    graph, changes = optimize(original)
    assert changes["original_operations"] == 2
    assert changes["final_operations"] == 1
    assert changes["fusion_patterns"] == 1
    assert graph.operations[0].op == "FusedGemmRelu"
    assert graph.operations[0].attrs["fused_from"] == [o.id for o in original.ordered()]
    assert "y" not in graph.tensors
    assert original.to_json() == before
    assert Graph.from_json(graph.to_json()).to_json() == graph.to_json()
    assert optimize(graph)[0].to_json() == graph.to_json()
    assert verify(path, graph)["passed"]
    h = Hardware(architecture=architecture, sram_bytes=64, tile_m=2, tile_n=3, tile_k=2)
    program = lower(graph, h)
    inputs = make_inputs(graph)
    np.testing.assert_allclose(
        replay_fused(program, graph, inputs), execute(graph, inputs)["out"], rtol=1e-4, atol=1e-5
    )
    assert all(i.get("tensor") != "y" for i in program.instructions)
    assert sum(i.get("elements", 0) for i in program.instructions) == 15 * (
        4 if bias is not None else 2
    )
    assert simulate(program)["peak_sram_bytes"] <= 64
    baseline, baseline_changes = optimize(original, False)
    assert baseline_changes["fusion_patterns"] == 0
    assert verify(path, baseline)["passed"]


@pytest.mark.parametrize("guard", ["public", "shared"])
def test_preserve_observable_and_shared_gemm_result(gemm_factory, guard):
    graph = import_onnx(gemm_factory(relu=True))
    if guard == "public":
        graph.outputs.append("y")
    else:
        graph.tensors["branch"] = Tensor("branch", graph.tensors["y"].shape)
        graph.operations.append(Operation("branch", "Relu", ["y"], "branch"))
        graph.outputs.append("branch")
    result, changes = optimize(graph)
    assert changes["fusion_patterns"] == 0
    assert not any(o.op == "FusedGemmRelu" for o in result.operations)
    inputs = make_inputs(graph)
    for tid, expected in execute(graph, inputs).items():
        np.testing.assert_array_equal(execute(result, inputs)[tid], expected)


def test_fused_output_can_have_multiple_consumers(model_factory):
    path = model_factory(
        [
            helper.make_node("Gemm", ["a", "b"], ["g"]),
            helper.make_node("Relu", ["g"], ["r"]),
            helper.make_node("Add", ["r", "r"], ["left"]),
            helper.make_node("Add", ["r", "r"], ["right"]),
        ],
        {"a": (2, 3), "b": (3, 4)},
        {"left": (2, 4), "right": (2, 4)},
    )
    graph, changes = optimize(import_onnx(path))
    assert changes["fusion_patterns"] == 1
    assert graph.operations[0].output == "r"
    assert verify(path, graph)["passed"]


def test_fusion_ids_do_not_collide(gemm_factory):
    graph = import_onnx(gemm_factory(relu=True))
    graph.operations[0].id = "dense"
    graph.operations[1].id = "activation"
    graph.tensors["side"] = Tensor("side", graph.tensors["a"].shape)
    graph.operations.append(Operation("dense_fused", "Relu", ["a"], "side"))
    graph.outputs.append("side")
    original = deepcopy(graph)
    result = fuse(graph)
    assert next(o.id for o in result.operations if o.op == "FusedGemmRelu") == "dense_fused_"
    assert graph.to_json() == original.to_json()
    result.validate()


@pytest.mark.parametrize(
    "architecture,baseline_cycles,fused_cycles,baseline_bytes,fused_bytes",
    [
        ("conventional", 59, 43, 184, 120),
        ("pim", 56, 40, 136, 72),
    ],
)
def test_hand_calculated_fused_timing(
    gemm_factory, architecture, baseline_cycles, fused_cycles, baseline_bytes, fused_bytes
):
    path = gemm_factory(
        attrs={"alpha": 2.0, "beta": -1.0}, bias_shape=(4,), m=2, k=3, n=4, relu=True
    )
    original = import_onnx(path)
    graph, _ = optimize(original)
    h = Hardware(
        architecture=architecture,
        macs_per_cycle=4,
        elements_per_cycle=4,
        dma_bytes_per_cycle=8,
        dma_setup_cycles=3,
        compute_setup_cycles=2,
        pim_macs_per_cycle=4,
        pim_bytes_per_cycle=8,
    )
    baseline = simulate(lower(original, h))
    fused = simulate(lower(graph, h))
    assert baseline["cycles"] == baseline_cycles
    assert fused["cycles"] == fused_cycles
    assert baseline["offchip_bytes"] == baseline_bytes
    assert fused["offchip_bytes"] == fused_bytes


@pytest.mark.parametrize("architecture", ["conventional", "pim"])
def test_relu_runs_after_reduction_scaling_and_bias(model_factory, architecture):
    path = model_factory(
        [
            helper.make_node("Gemm", ["a", "b", "c"], ["g"], alpha=-2.0),
            helper.make_node("Relu", ["g"], ["out"]),
        ],
        {"a": (1, 3), "b": (3, 1), "c": (1,)},
        {"out": (1, 1)},
    )
    graph, _ = optimize(import_onnx(path))
    inputs = {
        "a": np.array([[-3, 5, -1]], dtype=np.float32),
        "b": np.ones((3, 1), dtype=np.float32),
        "c": np.array([3], dtype=np.float32),
    }
    program = lower(graph, Hardware(architecture=architecture, tile_k=1, sram_bytes=32))
    np.testing.assert_array_equal(replay_fused(program, graph, inputs), np.array([[1]], np.float32))


def test_generated_gemm_relu_example(tmp_path):
    path = generate_model(tmp_path / "activated.onnx", "gemm_relu")
    graph = import_onnx(path)
    fused, changes = optimize(graph)
    assert changes["fusion_patterns"] == 1
    assert verify(path, fused)["passed"]


def test_fusion_comparison_report_and_reproducibility(tmp_path):
    runs = run_fusion_benchmarks(tmp_path)
    assert len(runs) == 4
    expected = {
        "conventional_unoptimized": 791,
        "conventional_fused": 675,
        "pim_unoptimized": 687,
        "pim_fused": 571,
    }
    for label, run in runs:
        assert run.metrics["cycles"] == expected[label]
        assert run.verification["passed"]
        assert run.changes["fusion_patterns"] == int(label.endswith("_fused"))
    report = render_report(runs, tmp_path / "report.html").read_text()
    assert "FusedGemmRelu" in report
    before = (tmp_path / "results.json").read_bytes()
    run_fusion_benchmarks(tmp_path)
    assert before == (tmp_path / "results.json").read_bytes()
