from dataclasses import asdict, replace

import numpy as np
import pytest
from onnx import helper

from npuforge import _native
from npuforge.backend import Hardware, broadcast_bytes, choose_tile, lower, simulate, tile_memory
from npuforge.compiler import optimize
from npuforge.functional import execute, make_inputs
from npuforge.importer import import_onnx
from npuforge.ir import Tensor
from npuforge.models import generate_model


@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("sram", [16, 48, 256, 4096])
def test_tiling_boundaries_and_mac_coverage(tmp_path, architecture, sram):
    graph = import_onnx(generate_model(tmp_path / "m.onnx", "matmul", 5, 7, 9))
    h = Hardware(architecture=architecture, sram_bytes=sram, tile_m=4, tile_n=6, tile_k=3)
    program = lower(graph, h)
    compute = [i for i in program.instructions if "MATMUL" in i["op"]]
    assert sum(i["macs"] for i in compute) == 5 * 7 * 9
    assert all(i.get("sram_bytes", 0) <= sram for i in program.instructions)
    stores = [i for i in program.instructions if i["op"] == "DMA_STORE"]
    assert sum(i["bytes"] for i in stores) == 5 * 9 * 4
    coverage = np.zeros((5, 9, 7), dtype=np.int8)
    for i in compute:
        r = i["ranges"]
        coverage[slice(*r["m"]), slice(*r["n"]), slice(*r["k"])] += 1
    assert np.all(coverage == 1)
    result = simulate(program)
    assert 0 < result["peak_sram_bytes"] <= sram
    assert result == simulate(program)
    for event in result["trace"]:
        deps = program.instructions[event["id"]]["deps"]
        ends = {e["id"]: e["end"] for e in result["trace"]}
        assert all(ends[d] <= event["start"] for d in deps)


def test_minimum_sram_and_bad_configs():
    with pytest.raises(ValueError, match="SRAM"):
        choose_tile(2, 3, 4, Hardware(sram_bytes=11))
    for params in (
        {"sram_bytes": 0},
        {"tile_k": -1},
        {"macs_per_cycle": 1.5},
        {"clock_hz": float("nan")},
        {"architecture": "other"},
        {"dma_setup_cycles": -1},
        {"tile_m": True},
    ):
        with pytest.raises(ValueError):
            Hardware(**params)
    assert tile_memory(1, 1, 1, True, False) == 16


def test_small_workload_hand_calculated(tmp_path):
    # One 2x3 @ 3x4 tile. 24 A bytes + 48 B bytes + 32 output bytes.
    graph = import_onnx(generate_model(tmp_path / "m.onnx", "matmul", 2, 3, 4))
    h = Hardware(
        macs_per_cycle=4, dma_bytes_per_cycle=8, dma_setup_cycles=3, compute_setup_cycles=2
    )
    result = simulate(lower(graph, h))
    # Loads 6+9, compute 2+6, store 3+4 = 30, peak 104.
    assert result["cycles"] == 30
    assert result["offchip_bytes"] == 104
    assert result["peak_sram_bytes"] == 104
    pim = simulate(
        lower(graph, replace(h, architecture="pim", pim_macs_per_cycle=4, pim_bytes_per_cycle=8))
    )
    # A load 6; PIM 2+6+6; store 7 => 27. B remains memory-local.
    assert pim["cycles"] == 27
    assert pim["offchip_bytes"] == 56
    assert pim["pim_local_read_bytes"] == 48
    assert pim["dram_read_bytes"] == 72


@pytest.mark.parametrize("a,b", [((2, 1, 3), (1, 4, 1)), ((), ()), ((2, 3), ())])
def test_elementwise_lowering(model_factory, a, b):
    shape = np.broadcast_shapes(a, b)
    graph = import_onnx(
        model_factory(
            [helper.make_node("Add", ["a", "b"], ["out"])], {"a": a, "b": b}, {"out": shape}
        )
    )
    h = Hardware(sram_bytes=48)
    program = lower(graph, h)
    assert sum(i.get("elements", 0) for i in program.instructions) == np.prod(shape)
    assert simulate(program)["peak_sram_bytes"] <= 48
    with pytest.raises(ValueError, match="SRAM"):
        lower(graph, replace(h, sram_bytes=4))


def test_broadcast_bytes_exact():
    assert broadcast_bytes(Tensor("a", (1, 4, 1)), (2, 4, 3), 0, 12) == 16
    assert broadcast_bytes(Tensor("a", ()), (2, 4, 3), 2, 8) == 4


def test_optimization_changes_real_transfers(tmp_path):
    graph = import_onnx(generate_model(tmp_path / "mlp.onnx"))
    optimized, _ = optimize(graph)
    baseline = simulate(lower(graph))
    fused = simulate(lower(optimized))
    assert fused["offchip_bytes"] < baseline["offchip_bytes"]
    assert len(lower(optimized).instructions) < len(lower(graph).instructions)


def native(program, **params):
    return _native.simulate(program, asdict(Hardware(**params)))


def instruction(i, op, deps=None, **kw):
    return {"id": i, "op": op, "deps": deps or [], **kw}


def test_independent_overlap_and_resource_contention():
    dma = instruction(0, "DMA_LOAD", bytes=64)
    compute = instruction(1, "MATMUL", macs=640)
    result = native([dma, compute])
    assert result["cycles"] == 14  # DMA 10 cycles overlaps compute 14.
    assert result["compute_utilization"] == 1
    assert result["dma_utilization"] == 10 / 14
    compute["deps"] = [0]
    assert native([dma, compute])["cycles"] == 24
    dma2 = instruction(1, "DMA_LOAD", bytes=64)
    assert native([dma, dma2])["cycles"] == 20


def test_sram_pressure_waits_until_release():
    instructions = [
        instruction(0, "ALLOC", buffer=0, sram_bytes=64),
        instruction(1, "ALLOC", buffer=1, sram_bytes=64),
        instruction(2, "DMA_LOAD", [0], buffer=0, bytes=64),
        instruction(3, "FREE", [2], buffer=0),
        instruction(4, "MATMUL", [1], buffer=1, macs=640),
        instruction(5, "FREE", [4], buffer=1),
    ]
    result = native(instructions, sram_bytes=64)
    assert result["cycles"] == 24
    assert result["peak_sram_bytes"] == 64
    assert native(instructions, sram_bytes=128)["cycles"] == 14


@pytest.mark.parametrize(
    "program",
    [
        [instruction(0, "unknown")],
        [instruction(1, "MATMUL")],
        [instruction(0, "MATMUL", [1]), instruction(1, "MATMUL", [0])],
        [instruction(0, "MATMUL", [3])],
        [instruction(0, "MATMUL", [0])],
        [instruction(0, "DMA_LOAD", buffer=0)],
        [instruction(0, "ALLOC", buffer=0, sram_bytes=65537)],
        [instruction(0, "ALLOC", buffer=0, sram_bytes=32)],
        [instruction(0, "FREE", buffer=0)],
    ],
)
def test_native_rejects_invalid_programs(program):
    with pytest.raises(ValueError):
        native(program)


def test_bounded_random_workloads(tmp_path):
    rng = np.random.default_rng(33)
    for index in range(12):
        dims = [int(x) for x in rng.integers(1, 14, size=3)]
        graph = import_onnx(generate_model(tmp_path / f"random{index}.onnx", "mlp", *dims))
        optimized, _ = optimize(graph, index % 2 == 0)
        h = Hardware(
            architecture="pim" if index % 3 == 0 else "conventional",
            sram_bytes=int(rng.choice([64, 128, 1024])),
        )
        result = simulate(lower(optimized, h))
        assert result["peak_sram_bytes"] <= h.sram_bytes
        assert 0 <= result["dram_utilization"] <= 1
        assert result["cycles"] > 0


def test_transfer_and_compute_cannot_exceed_allocated_buffer():
    for op, parameters in (
        ("DMA_LOAD", {"bytes": 64}),
        ("MATMUL", {"macs": 1, "working_set_bytes": 64}),
    ):
        program = [
            instruction(0, "ALLOC", buffer=0, sram_bytes=32),
            instruction(1, op, [0], buffer=0, **parameters),
            instruction(2, "FREE", [1], buffer=0),
        ]
        with pytest.raises(ValueError, match="working set"):
            native(program)


@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("sram", [64, 256])
def test_lowered_tile_arithmetic_against_graph(tmp_path, architecture, sram):
    graph, _ = optimize(import_onnx(generate_model(tmp_path / "m.onnx", "mlp", 5, 7, 9)))
    inputs = make_inputs(graph)
    values = {k: t.data for k, t in graph.tensors.items() if t.data is not None}
    values.update(inputs)
    for node in graph.operations:
        values[node.output] = np.zeros(graph.tensors[node.output].shape, dtype=np.float32)
    h = Hardware(architecture=architecture, sram_bytes=sram, tile_m=3, tile_n=4, tile_k=2)
    accumulators = {}
    for i in lower(graph, h).instructions:
        op = i["op"]
        if "MATMUL" in op:
            r = i["ranges"]
            ms, ns, ks = (slice(*r[axis]) for axis in ("m", "n", "k"))
            partial = values[i["inputs"][0]][ms, ks] @ values[i["inputs"][1]][ks, ns]
            acc = accumulators[i["buffer"]] + partial if i["accumulate"] else partial
            if i["fused_epilogue"]:
                bias = np.broadcast_to(values[i["inputs"][2]], values[i["output"]].shape)
                acc = np.maximum(acc + bias[ms, ns], np.float32(0))
            accumulators[i["buffer"]] = acc
        elif op in {"ADD", "RELU"}:
            shape = values[i["output"]].shape
            start, stop = i["flat_range"]
            args = [np.broadcast_to(values[t], shape).reshape(-1)[start:stop] for t in i["inputs"]]
            accumulators[i["buffer"]] = args[0] + args[1] if op == "ADD" else np.maximum(args[0], 0)
        elif op == "DMA_STORE":
            if "ranges" in i:
                ms, ns = (slice(*i["ranges"][axis]) for axis in ("m", "n"))
                values[i["output"]][ms, ns] = accumulators[i["buffer"]]
            else:
                start, stop = i["flat_range"]
                values[i["output"]].reshape(-1)[start:stop] = accumulators[i["buffer"]]
    for tid, reference in execute(graph, inputs).items():
        np.testing.assert_allclose(values[tid], reference, rtol=1e-4, atol=1e-5)


def test_random_instruction_contention_and_durations():
    rng = np.random.default_rng(21)
    instructions = []
    for index in range(48):
        op = str(rng.choice(["DMA_LOAD", "DMA_STORE", "MATMUL", "PIM_MATMUL"]))
        deps = sorted(set(int(x) for x in rng.integers(0, index, size=2))) if index else []
        instructions.append(
            instruction(
                index,
                op,
                deps,
                bytes=int(rng.integers(1, 100)),
                macs=int(rng.integers(1, 300)),
                local_read_bytes=64,
            )
        )
    result = native(instructions)
    assert result == native(instructions)
    events = {e["id"]: e for e in result["trace"]}
    resources = {
        "DMA_LOAD": {"dma", "dram"},
        "DMA_STORE": {"dma", "dram"},
        "MATMUL": {"compute"},
        "PIM_MATMUL": {"pim", "dram"},
    }
    for i in instructions:
        e = events[i["id"]]
        assert all(events[dep]["end"] <= e["start"] for dep in i["deps"])
        if i["op"].startswith("DMA"):
            expected = 8 + (i["bytes"] + 31) // 32
        elif i["op"] == "MATMUL":
            expected = 4 + (i["macs"] + 63) // 64
        else:
            expected = 4 + (i["macs"] + 63) // 64 + 1
        assert e["end"] - e["start"] == expected
        for other in instructions[: i["id"]]:
            if resources[i["op"]] & resources[other["op"]]:
                f = events[other["id"]]
                assert e["end"] <= f["start"] or f["end"] <= e["start"]
    assert result["cycles"] == max(e["end"] for e in result["trace"])


def test_clock_scaling_and_forward_dependencies():
    program = [instruction(0, "MATMUL", [1], macs=640), instruction(1, "DMA_LOAD", bytes=64)]
    result = native(program, clock_hz=500_000_000)
    assert result["cycles"] == 24
    assert result["latency_ns"] == 48
