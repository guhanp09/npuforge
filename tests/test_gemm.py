"""Gemm import, functional, lowering and timing checks independent of ORT execution."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from onnx import helper

from npuforge.backend import Hardware, lower, simulate
from npuforge.compiler import optimize
from npuforge.functional import execute, make_inputs, verify
from npuforge.importer import import_onnx
from npuforge.ir import Graph, Operation, Tensor
from npuforge.models import generate_model
from npuforge.pipeline import compile_model


@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("bias_shape", [None, (), (1,), (5,), (1, 5), (3, 1), (1, 1), (3, 5)])
def test_gemm_transposes_and_broadcasts(gemm_factory, trans_a, trans_b, bias_shape):
    path = gemm_factory(
        attrs={"transA": trans_a, "transB": trans_b, "alpha": 0.75, "beta": -1.5},
        bias_shape=bias_shape,
    )
    graph = import_onnx(path)
    assert graph.operations[0].op == "Gemm"
    assert graph.tensors["y"].shape == (3, 5)
    assert graph.operations[0].attrs["alpha"] == 0.75
    assert verify(path, graph)["passed"]
    assert Graph.from_json(graph.to_json()).to_json() == graph.to_json()


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.0, 0.0), (-2.0, 0.0), (0.0, 2.0)])
def test_gemm_scalars_and_default_attributes(gemm_factory, alpha, beta):
    attrs = {} if alpha == beta == 1 else {"alpha": alpha, "beta": beta}
    path = gemm_factory(attrs=attrs, bias_shape=(5,), constants=True)
    graph = import_onnx(path)
    assert verify(path, graph)["passed"]
    if not attrs:
        assert graph.operations[0].attrs == {}


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_gemm_missing_optional_c(gemm_factory, explicit_empty):
    path = gemm_factory(attrs={"alpha": -0.5, "beta": 7.0}, explicit_empty=explicit_empty)
    graph = import_onnx(path)
    assert graph.operations[0].inputs == ["a", "b"]
    assert verify(path, graph)["passed"]
    compute = [i for i in lower(graph).instructions if i["op"] == "MATMUL"]
    assert sum(i["elements"] for i in compute) == 15  # Alpha only; beta has no input.


def test_nonzero_transpose_flags(gemm_factory):
    path = gemm_factory(attrs={"transA": -3, "transB": 2}, bias_shape=(5,))
    assert verify(path, import_onnx(path))["passed"]


@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("bias_shape", [None, (), (5,), (3, 1), (3, 5)])
@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_gemm_lowered_tile_semantics(gemm_factory, architecture, bias_shape, trans_a, trans_b):
    path = gemm_factory(
        attrs={"transA": trans_a, "transB": trans_b, "alpha": -0.5, "beta": 2.0},
        bias_shape=bias_shape,
        constants=True,
        relu=True,
    )
    graph = import_onnx(path)
    inputs = make_inputs(graph)
    values = {tid: t.data for tid, t in graph.tensors.items() if t.data is not None}
    values.update(inputs)
    values["y"] = np.zeros((3, 5), dtype=np.float32)
    h = Hardware(architecture=architecture, sram_bytes=64, tile_m=2, tile_n=3, tile_k=2)
    program = lower(graph, h)
    accumulators = {}
    staging = {}
    macs, scalar_work = 0, 0
    for i in program.instructions:
        if i["op"] == "DMA_LOAD" and "source_ranges" in i:
            r = i["source_ranges"]
            tile = values[i["tensor"]][slice(*r["rows"]), slice(*r["cols"])]
            assert tile.nbytes == i["bytes"]
            if i["transpose"]:
                tile = tile.T
            expected = (
                (
                    i["ranges"]["m"][1] - i["ranges"]["m"][0],
                    i["ranges"]["k"][1] - i["ranges"]["k"][0],
                )
                if i["tensor"] == "a"
                else (
                    i["ranges"]["k"][1] - i["ranges"]["k"][0],
                    i["ranges"]["n"][1] - i["ranges"]["n"][0],
                )
            )
            assert tile.shape == expected
            staging[i["buffer"], i["tensor"]] = tile
        elif i["op"] == "DMA_LOAD" and i["tensor"] == "c":
            ms, ns = (slice(*i["ranges"][axis]) for axis in ("m", "n"))
            positions = np.arange(np.prod(bias_shape), dtype=np.int64).reshape(bias_shape)
            unique = np.unique(np.broadcast_to(positions, (3, 5))[ms, ns])
            assert i["bytes"] == unique.size * 4
        elif i["op"] in {"MATMUL", "PIM_MATMUL"}:
            r = i["ranges"]
            ms, ns = (slice(*r[axis]) for axis in ("m", "n"))
            if i["op"] == "PIM_MATMUL":
                source = i["local_source_ranges"]
                weight = values["b"][slice(*source["rows"]), slice(*source["cols"])]
                assert weight.nbytes == i["local_read_bytes"]
                if i["local_transpose"]:
                    weight = weight.T
            else:
                weight = staging[i["buffer"], "b"]
            # Reconstruct values from the actual DMA/local-read metadata, rather
            # than executing the original Gemm node as a substitute for lowering.
            partial = staging[i["buffer"], "a"] @ weight
            acc = accumulators[i["buffer"]] + partial if i["accumulate"] else partial
            if i["gemm_epilogue"]:
                acc = np.float32(-0.5) * acc
                if bias_shape is not None:
                    bias = np.broadcast_to(values["c"], (3, 5))[ms, ns]
                    acc = acc + np.float32(2.0) * bias
            else:
                assert i["elements"] == 0
            accumulators[i["buffer"]] = acc
            macs += i["macs"]
            scalar_work += i["elements"]
        elif i["op"] == "DMA_STORE" and i["output"] == "y":
            ms, ns = (slice(*i["ranges"][axis]) for axis in ("m", "n"))
            values["y"][ms, ns] = accumulators[i["buffer"]]
    assert macs == 3 * 7 * 5
    assert scalar_work == 15 * (3 if bias_shape is not None else 1)
    values["out"] = np.maximum(values["y"], np.float32(0))
    np.testing.assert_allclose(values["out"], execute(graph, inputs)["out"], rtol=1e-4, atol=1e-5)
    for enabled in (False, True):
        run = compile_model(path, h, enabled)
        assert run.verification["passed"]
        assert run.metrics["peak_sram_bytes"] <= h.sram_bytes


def test_gemm_hand_calculated_native_timing(gemm_factory):
    path = gemm_factory(attrs={"alpha": 2.0, "beta": -1.0}, bias_shape=(4,), m=2, k=3, n=4)
    graph = import_onnx(path)
    h = Hardware(
        macs_per_cycle=4,
        elements_per_cycle=4,
        dma_bytes_per_cycle=8,
        dma_setup_cycles=3,
        compute_setup_cycles=2,
    )
    conventional = simulate(lower(graph, h))
    # Loads A 6 + B 9 + C 5; MAC 2+6, epilogue 24/4=6; store 7 => 41.
    assert conventional["cycles"] == 41
    assert conventional["offchip_bytes"] == 120
    assert conventional["peak_sram_bytes"] == 136
    pim = simulate(
        lower(graph, replace(h, architecture="pim", pim_macs_per_cycle=4, pim_bytes_per_cycle=8))
    )
    # A 6 + C 5; PIM 2+6+6+6; store 7 => 38, no B SRAM staging.
    assert pim["cycles"] == 38
    assert pim["offchip_bytes"] == 72
    assert pim["pim_local_read_bytes"] == 48
    assert pim["peak_sram_bytes"] == 88


@pytest.mark.parametrize(
    "mutation",
    ["rank", "reduction", "bias_expand", "bias_rank", "alpha", "beta", "flag", "unknown", "arity"],
)
def test_invalid_gemm_ir_is_rejected(mutation):
    graph = Graph(
        {
            "a": Tensor("a", (2, 3)),
            "b": Tensor("b", (3, 4)),
            "c": Tensor("c", (4,)),
            "y": Tensor("y", (2, 4)),
        },
        [Operation("gemm", "Gemm", ["a", "b", "c"], "y")],
        ["a", "b", "c"],
        ["y"],
    )
    if mutation == "rank":
        graph.tensors["a"].shape = (3,)
    elif mutation == "reduction":
        graph.tensors["b"].shape = (5, 4)
    elif mutation == "bias_expand":
        graph.tensors["c"].shape = (3, 2, 4)
    elif mutation == "bias_rank":
        graph.tensors["c"].shape = (1, 2, 4)
    elif mutation in {"alpha", "beta"}:
        graph.operations[0].attrs[mutation] = float("nan")
    elif mutation == "flag":
        graph.operations[0].attrs["transA"] = 1.0
    elif mutation == "unknown":
        graph.operations[0].attrs["broadcast"] = 1
    else:
        graph.operations[0].inputs = ["a"]
    with pytest.raises(ValueError):
        graph.validate()


def test_gemm_optimization_preserves_dependencies(model_factory):
    nodes = [
        helper.make_node("Gemm", ["a", "b", "c"], ["live"], alpha=0.5),
        helper.make_node("Gemm", ["a", "b"], ["dead"]),
        helper.make_node("Relu", ["live"], ["y"]),
        helper.make_node("Add", ["live", "y"], ["out"]),
    ]
    path = model_factory(nodes, {"a": (2, 3), "b": (3, 4), "c": (4,)}, {"out": (2, 4)})
    graph = import_onnx(path)
    original = deepcopy(graph)
    optimized, changes = optimize(graph)
    assert changes["dead_operations_removed"] == 1
    assert [op.op for op in optimized.ordered()] == ["Gemm", "Relu", "Add"]
    assert graph.to_json() == original.to_json()
    assert verify(path, optimized)["passed"]
    program = lower(optimized, Hardware(sram_bytes=64))
    assert simulate(program)["cycles"] > 0


@pytest.mark.parametrize(
    "bias_shape,attrs",
    [
        ((3, 5), {"alpha": float("inf")}),
        ((3, 5), {"beta": float("nan")}),
        ((3, 5), {"transA": 1.0}),
        ((2, 5), {}),
        ((3, 5), {"broadcast": 1}),
    ],
)
def test_invalid_gemm_onnx_rejected(gemm_factory, bias_shape, attrs):
    with pytest.raises(ValueError):
        import_onnx(gemm_factory(bias_shape=bias_shape, attrs=attrs))


def test_generated_gemm_cli_report(tmp_path):
    path = generate_model(tmp_path / "dense.onnx", "gemm")
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npuforge.cli",
            "compile",
            str(path),
            "--compare",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[PASS] Native C++ simulation" in result.stdout
    report = (output / "report.html").read_text()
    assert "Gemm" in report
    assert "alpha" in report
    for architecture in ("conventional", "pim"):
        summary = json.loads((output / architecture / "summary.json").read_text())
        assert summary["verification"]["passed"]
        ir = Graph.from_json((output / architecture / "original.ir.json").read_text())
        assert ir.operations[0].attrs == {"transB": 1, "alpha": 0.75, "beta": 0.5}
