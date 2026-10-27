"""The public end-to-end compilation API and inspectable run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .backend import Hardware, Program, lower, simulate
from .compiler import optimize
from .functional import verify
from .importer import import_onnx
from .ir import Graph


@dataclass
class Run:
    model: str
    original: Graph
    graph: Graph
    changes: dict
    program: Program
    metrics: dict
    verification: dict

    def summary(self) -> dict:
        return {
            "model": self.model,
            "architecture": self.program.hardware.architecture,
            "transformations": self.changes,
            "verification": self.verification,
            "metrics": {k: v for k, v in self.metrics.items() if k != "trace"},
        }

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "original.ir.json").write_text(self.original.to_json())
        (directory / "optimized.ir.json").write_text(self.graph.to_json())
        (directory / "instructions.json").write_text(self.program.to_json())
        for name, data in [("simulation", self.metrics), ("summary", self.summary())]:
            (directory / f"{name}.json").write_text(
                json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )


def compile_model(
    path: str | Path, hardware: Hardware | None = None, optimizations: bool = True
) -> Run:
    original = import_onnx(path)
    graph, changes = optimize(original, optimizations)
    program = lower(graph, hardware)
    metrics = simulate(program)
    verification = verify(path, graph)
    return Run(Path(path).name, original, graph, changes, program, metrics, verification)
