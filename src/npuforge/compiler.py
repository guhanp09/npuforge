"""Two semantics-preserving IR passes with explicit observability safeguards."""

from copy import deepcopy

from .ir import Graph, Operation


def eliminate_dead(graph: Graph) -> Graph:
    graph.validate()
    result = deepcopy(graph)
    producers = result.producers()
    needed = set(result.outputs)
    stack = list(needed)
    keep = set()
    while stack:
        tid = stack.pop()
        if tid in producers:
            op = producers[tid]
            keep.add(op.id)
            for source in op.inputs:
                if source not in needed:
                    needed.add(source)
                    stack.append(source)
    result.operations = [op for op in result.ordered() if op.id in keep]
    # Keep the public input contract, including unused inputs for ORT verification.
    result.tensors = {k: t for k, t in result.tensors.items() if k in needed or k in result.inputs}
    result.validate()
    return result


def fuse(graph: Graph) -> Graph:
    graph.validate()
    result = deepcopy(graph)
    consumers = result.consumers()
    removed = set()
    replacements = {}
    used_ids = {o.id for o in result.operations}

    def fused_id(source: str) -> str:
        candidate = source + "_fused"
        while candidate in used_ids:
            candidate += "_"
        used_ids.add(candidate)
        return candidate

    for mm in result.ordered():
        if mm.op == "Gemm" and mm.output not in result.outputs:
            users = consumers[mm.output]
            if len(users) == 1 and users[0].op == "Relu":
                relu = users[0]
                replacements[relu.id] = Operation(
                    fused_id(mm.id),
                    "FusedGemmRelu",
                    list(mm.inputs),
                    relu.output,
                    mm.attrs | {"fused_from": [mm.id, relu.id]},
                )
                removed.add(mm.id)
            continue
        if mm.op != "MatMul" or mm.output in result.outputs:
            continue
        users = consumers[mm.output]
        if len(users) != 1 or users[0].op != "Add":
            continue
        add = users[0]
        # x + x is not a bias epilogue; it requires the intermediate value twice.
        if add.inputs.count(mm.output) != 1 or add.output in result.outputs:
            continue
        bias = next(t for t in add.inputs if t != mm.output)
        if result.tensors[add.output].shape != result.tensors[mm.output].shape:
            continue
        users = consumers[add.output]
        if len(users) != 1 or users[0].op != "Relu":
            continue
        relu = users[0]
        replacements[relu.id] = Operation(
            fused_id(mm.id),
            "FusedMatMulAddRelu",
            [*mm.inputs, bias],
            relu.output,
            {"fused_from": [mm.id, add.id, relu.id]},
        )
        removed.update([mm.id, add.id])
    result.operations = [replacements.get(o.id, o) for o in result.ordered() if o.id not in removed]
    # Prune the now-unproduced intermediate tensors, without removing public inputs.
    used = set(result.inputs + result.outputs)
    for op in result.operations:
        used.update(op.inputs + [op.output])
    result.tensors = {k: t for k, t in result.tensors.items() if k in used}
    result.validate()
    return result


def optimize(graph: Graph, enabled: bool = True) -> tuple[Graph, dict]:
    graph.validate()
    dead = eliminate_dead(graph) if enabled else deepcopy(graph)
    result = fuse(dead) if enabled else dead
    return result, {
        "enabled": enabled,
        "original_operations": len(graph.operations),
        "dead_operations_removed": len(graph.operations) - len(dead.operations),
        "fusion_patterns": sum(
            o.op in {"FusedMatMulAddRelu", "FusedGemmRelu"} for o in result.operations
        ),
        "final_operations": len(result.operations),
    }
