"""Scratchpad-bounded tiling and explicit data movement into a native instruction DAG."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .ir import Graph, Tensor, gemm_attributes


@dataclass(frozen=True)
class Hardware:
    architecture: str = "conventional"
    sram_bytes: int = 65536
    macs_per_cycle: int = 64
    elements_per_cycle: int = 16
    dma_bytes_per_cycle: int = 32
    dma_setup_cycles: int = 8
    compute_setup_cycles: int = 4
    pim_macs_per_cycle: int = 64
    pim_bytes_per_cycle: int = 128
    clock_hz: float = 1e9
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16

    def __post_init__(self) -> None:
        if self.architecture not in {"conventional", "pim"}:
            raise ValueError("architecture must be conventional or pim")
        for key, value in asdict(self).items():
            if key in {"architecture", "clock_hz"}:
                continue
            minimum = 0 if key.endswith("setup_cycles") else 1
            if type(value) is not int or value < minimum or value > 2**53:
                raise ValueError(f"{key}: expected integer in [{minimum}, 2**53]")
        if (
            isinstance(self.clock_hz, bool)
            or not isinstance(self.clock_hz, (int, float))
            or not math.isfinite(self.clock_hz)
            or self.clock_hz <= 0
        ):
            raise ValueError("clock_hz must be finite and positive")

    @classmethod
    def load(cls, path: str | Path) -> Hardware:
        try:
            return cls(**json.loads(Path(path).read_text()))
        except TypeError as exc:
            raise ValueError(f"invalid hardware configuration: {exc}") from exc

    def alternative(self, architecture: str) -> Hardware:
        return replace(self, architecture=architecture)


def tile_memory(m: int, n: int, k: int, fused: bool, pim: bool) -> int:
    # A staging + B staging (conventional only) + FP32 accumulator + optional bias staging.
    return 4 * (m * k + (0 if pim else k * n) + m * n * (2 if fused else 1))


def choose_tile(m: int, n: int, k: int, h: Hardware, fused: bool = False) -> tuple[int, int, int]:
    dims = [min(m, h.tile_m), min(n, h.tile_n), min(k, h.tile_k)]
    while tile_memory(*dims, fused, h.architecture == "pim") > h.sram_bytes:
        candidates = [i for i, d in enumerate(dims) if d > 1]
        if not candidates:
            raise ValueError("SRAM too small for one legal MatMul tile; increase sram_bytes")
        # Choose the reduction with the largest actual memory saving.
        axis = max(
            candidates,
            key=lambda i: (
                tile_memory(
                    *[max(1, d // 2) if j == i else d for j, d in enumerate(dims)],
                    fused,
                    h.architecture == "pim",
                )
                * -1
            ),
        )
        dims[axis] = max(1, dims[axis] // 2)
    return tuple(dims)


def broadcast_bytes(tensor: Tensor, shape: tuple[int, ...], start: int, stop: int) -> int:
    """Unique source values for a contiguous flattened output tile (including scalars)."""
    if not shape:
        return 4
    coords = np.unravel_index(np.arange(start, stop, dtype=np.int64), shape)
    aligned = (1,) * (len(shape) - len(tensor.shape)) + tensor.shape
    indices = np.zeros(stop - start, dtype=np.int64)
    for axis, dim in enumerate(aligned):
        indices *= dim
        if dim != 1:
            indices += coords[axis]
    return int(np.unique(indices).size) * 4


@dataclass
class Program:
    instructions: list[dict]
    hardware: Hardware
    tiles: list[dict]

    def to_dict(self) -> dict:
        return {
            "schema": "npuforge.instructions.v1",
            "hardware": asdict(self.hardware),
            "tiles": self.tiles,
            "instructions": self.instructions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


def lower(graph: Graph, hardware: Hardware | None = None) -> Program:
    graph.validate()
    h = hardware or Hardware()
    instructions: list[dict] = []
    tiles: list[dict] = []
    produced: dict[str, list[int]] = {}
    last_release: list[int] = []
    buffer = 0

    def emit(op: str, deps: list[int], **kwargs) -> int:
        iid = len(instructions)
        instructions.append(
            {"id": iid, "op": op, "deps": sorted(set(deps)), "buffer": buffer, **kwargs}
        )
        return iid

    for node in graph.ordered():
        node_deps = [i for tid in node.inputs for i in produced.get(tid, [])]
        stores = []
        common = {"node": node.id, "inputs": node.inputs, "output": node.output}
        if node.op in {"MatMul", "Gemm", "FusedMatMulAddRelu", "FusedGemmRelu"}:
            gemm = gemm_attributes(node) if node.op in {"Gemm", "FusedGemmRelu"} else None
            gemm_relu = node.op == "FusedGemmRelu"
            a_shape = graph.tensors[node.inputs[0]].shape
            b_shape = graph.tensors[node.inputs[1]].shape
            m, k = a_shape[::-1] if gemm and gemm["transA"] else a_shape
            _, n = b_shape[::-1] if gemm and gemm["transB"] else b_shape
            if gemm:
                common["gemm"] = gemm
            fused = node.op == "FusedMatMulAddRelu"
            bias_staging = fused or (gemm is not None and len(node.inputs) == 3)
            pim = h.architecture == "pim"
            tm, tn, tk = choose_tile(m, n, k, h, bias_staging)
            tiles.append({"node": node.id, "tile": [tm, tn, tk], "matrix": [m, n, k]})
            for mi in range(0, m, tm):
                a = min(tm, m - mi)
                for ni in range(0, n, tn):
                    b = min(tn, n - ni)
                    memory = tile_memory(a, b, min(tk, k), bias_staging, pim)
                    alloc = emit("ALLOC", node_deps + last_release, sram_bytes=memory, **common)
                    previous = [alloc]
                    for ki in range(0, k, tk):
                        c = min(tk, k - ki)
                        ranges = {"m": [mi, mi + a], "n": [ni, ni + b], "k": [ki, ki + c]}
                        a_source = (
                            {"rows": ranges["k"], "cols": ranges["m"]}
                            if (gemm and gemm["transA"])
                            else {"rows": ranges["m"], "cols": ranges["k"]}
                        )
                        b_source = (
                            {"rows": ranges["n"], "cols": ranges["k"]}
                            if (gemm and gemm["transB"])
                            else {"rows": ranges["k"], "cols": ranges["n"]}
                        )
                        load_a = emit(
                            "DMA_LOAD",
                            previous,
                            bytes=4 * a * c,
                            tensor=node.inputs[0],
                            source_ranges=a_source,
                            transpose=bool(gemm and gemm["transA"]),
                            ranges=ranges,
                            **common,
                        )
                        compute_deps = [load_a]
                        if not pim:
                            load_b = emit(
                                "DMA_LOAD",
                                previous,
                                bytes=4 * b * c,
                                tensor=node.inputs[1],
                                source_ranges=b_source,
                                transpose=bool(gemm and gemm["transB"]),
                                ranges=ranges,
                                **common,
                            )
                            compute_deps.append(load_b)
                        final = ki + c == k
                        # Fill bias staging only for the final reduction chunk.
                        if bias_staging and final:
                            bias = graph.tensors[node.inputs[2]]
                            aligned = (1,) * (2 - len(bias.shape)) + bias.shape
                            bias_bytes = (
                                4 * (a if aligned[0] != 1 else 1) * (b if aligned[1] != 1 else 1)
                            )
                            compute_deps.append(
                                emit(
                                    "DMA_LOAD",
                                    previous,
                                    bytes=bias_bytes,
                                    tensor=node.inputs[2],
                                    ranges=ranges,
                                    **common,
                                )
                            )
                        opcode = (
                            "PIM_MATMUL"
                            if pim
                            else ("FUSED_MATMUL_ADD_RELU" if fused and final else "MATMUL")
                        )
                        epilogue_elements = 2 * a * b if fused else 0
                        if gemm:
                            scalar_ops = int(gemm["alpha"] != 1) + int(gemm_relu)
                            if bias_staging:
                                scalar_ops += 1 + int(gemm["beta"] != 1)
                            epilogue_elements = scalar_ops * a * b
                        compute = emit(
                            opcode,
                            compute_deps,
                            macs=a * b * c,
                            working_set_bytes=memory,
                            elements=epilogue_elements if final else 0,
                            local_read_bytes=4 * b * c if pim else 0,
                            **(
                                {
                                    "local_source_ranges": b_source,
                                    "local_transpose": bool(gemm and gemm["transB"]),
                                }
                                if pim
                                else {}
                            ),
                            ranges=ranges,
                            accumulate=ki != 0,
                            fused_epilogue=fused and final,
                            **(
                                {"gemm_epilogue": final, "gemm_relu": gemm_relu and final}
                                if gemm
                                else {}
                            ),
                            **common,
                        )
                        previous = [compute]
                    store = emit(
                        "DMA_STORE",
                        previous,
                        bytes=4 * a * b,
                        tensor=node.output,
                        ranges={"m": [mi, mi + a], "n": [ni, ni + b]},
                        **common,
                    )
                    stores.append(store)
                    last_release = [emit("FREE", [store], **common)]
                    buffer += 1
        else:
            shape = graph.tensors[node.output].shape
            count = math.prod(shape)
            # Inputs are expanded in SRAM, even when unique DRAM source values broadcast.
            per_element = 4 * (len(node.inputs) + 1)
            chunk = min(8192, h.sram_bytes // per_element)
            if chunk < 1:
                raise ValueError(f"SRAM too small for {node.op}; need {per_element} bytes")
            for start in range(0, count, chunk):
                stop = min(count, start + chunk)
                alloc = emit(
                    "ALLOC",
                    node_deps + last_release,
                    sram_bytes=(stop - start) * per_element,
                    **common,
                )
                loads = [
                    emit(
                        "DMA_LOAD",
                        [alloc],
                        bytes=broadcast_bytes(graph.tensors[tid], shape, start, stop),
                        tensor=tid,
                        flat_range=[start, stop],
                        **common,
                    )
                    for tid in node.inputs
                ]
                compute = emit(
                    "ADD" if node.op == "Add" else "RELU",
                    loads,
                    elements=stop - start,
                    working_set_bytes=(stop - start) * per_element,
                    flat_range=[start, stop],
                    **common,
                )
                store = emit(
                    "DMA_STORE",
                    [compute],
                    bytes=4 * (stop - start),
                    tensor=node.output,
                    flat_range=[start, stop],
                    **common,
                )
                stores.append(store)
                last_release = [emit("FREE", [store], **common)]
                buffer += 1
        produced[node.output] = stores
    return Program(instructions, h, tiles)


def simulate(program: Program) -> dict:
    from . import _native

    result = _native.simulate(program.instructions, asdict(program.hardware))
    # Busy times sum instruction durations; elapsed time is the scheduler makespan.
    timing = {}
    for event in result["trace"]:
        node = program.instructions[event["id"]].get("node", "unknown")
        if node not in timing:
            timing[node] = {"start": event["start"], "end": event["end"], "busy_cycles": 0}
        timing[node]["start"] = min(timing[node]["start"], event["start"])
        timing[node]["end"] = max(timing[node]["end"], event["end"])
        timing[node]["busy_cycles"] += event["end"] - event["start"]
    result["per_operation"] = timing
    return result
