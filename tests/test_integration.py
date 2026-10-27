import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from npuforge.backend import Hardware, lower, simulate
from npuforge.benchmark import run_benchmarks
from npuforge.compiler import optimize
from npuforge.functional import verify
from npuforge.importer import import_onnx
from npuforge.models import generate_model
from npuforge.pipeline import compile_model
from npuforge.reports import graph_svg, render_report, timeline_svg


def test_cli_demo_artifacts_and_reproducibility(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "npuforge.cli", "demo", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[PASS] Native C++" in result.stdout
    assert "Report:" in result.stdout
    report = (tmp_path / "report.html").read_text()
    assert report.startswith("<!doctype html>")
    assert "<script" not in report
    assert 'src="http' not in report
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == 6
    for row in rows:
        assert row["verification"]["passed"]
        assert f"{row['metrics']['cycles']:,}" in report
        run_dir = tmp_path / row["case"]
        program = json.loads((run_dir / "instructions.json").read_text())
        assert program["schema"] == "npuforge.instructions.v1"
        assert (run_dir / "optimized.ir.json").exists()
    before = (tmp_path / "results.json").read_bytes()
    run_benchmarks(tmp_path)
    assert (tmp_path / "results.json").read_bytes() == before


def test_cli_commands_and_actionable_errors(tmp_path):
    path = generate_model(tmp_path / "m.onnx", "matmul")
    for args in (
        ["inspect", str(path)],
        ["inspect", str(path), "--json"],
        [
            "compile",
            str(path),
            "--compare",
            "--no-opt",
            "--arch",
            "pim",
            "--output",
            str(tmp_path / "custom"),
        ],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "npuforge.cli", *args], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    bad = subprocess.run(
        [sys.executable, "-m", "npuforge.cli", "compile", "missing.onnx"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "npuforge:" in bad.stderr
    assert "Traceback" not in bad.stderr
    config = tmp_path / "bad.json"
    config.write_text('{"sram_bytes": 0}')
    bad = subprocess.run(
        [sys.executable, "-m", "npuforge.cli", "demo", "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "sram_bytes" in bad.stderr


def test_report_escaping_metrics_and_svg(tmp_path):
    path = generate_model(tmp_path / "m.onnx")
    run = compile_model(path)
    run.model = '<img src=x onerror="alert(1)">'
    run.graph.name = "<script>"
    report = render_report([("custom <model>", run)], tmp_path / "r.html").read_text()
    assert "<img src=x" not in report
    assert "&lt;img" in report
    assert "custom &lt;model&gt;" in report
    assert f"{run.metrics['cycles']:,}" in report
    for svg in (timeline_svg(run), graph_svg(run.graph)):
        ET.fromstring(svg)
    assert run.metrics["latency_ns"] == run.metrics["cycles"]
    assert run.metrics["offchip_bytes"] == (
        run.metrics["offchip_read_bytes"] + run.metrics["dram_write_bytes"]
    )
    with pytest.raises(ValueError):
        render_report([], tmp_path / "empty.html")


def test_random_valid_graph_dependencies(model_factory):
    rng = np.random.default_rng(100)
    for case in range(10):
        nodes, available = [], ["x"]
        for index in range(8):
            out = f"t{index}"
            if rng.random() < 0.5:
                inputs = [available[int(rng.integers(len(available)))]]
                op = "Relu"
            else:
                inputs = [available[int(rng.integers(len(available)))] for _ in range(2)]
                op = "Add"
            nodes.append(helper.make_node(op, inputs, [out]))
            available.append(out)
        path = model_factory(
            nodes, {"x": (3, 5)}, {"t7": (3, 5), "t3": (3, 5)}, name=f"random_dag{case}"
        )
        graph = import_onnx(path)
        optimized, _ = optimize(graph)
        assert verify(path, optimized, seed=case)["passed"]
        result = simulate(lower(optimized, Hardware(sram_bytes=64)))
        assert result["peak_sram_bytes"] <= 64


def test_config_files():
    root = Path(__file__).resolve().parents[1]
    for arch in ("conventional", "pim"):
        assert Hardware.load(root / "configs" / f"{arch}.json").architecture == arch


def test_corrupt_onnx_is_an_actionable_cli_error(tmp_path):
    path = tmp_path / "corrupt.onnx"
    path.write_bytes(b"this is not a protobuf model")
    bad = subprocess.run(
        [sys.executable, "-m", "npuforge.cli", "compile", str(path)], capture_output=True, text=True
    )
    assert bad.returncode == 2
    assert "invalid ONNX model" in bad.stderr
    assert "Traceback" not in bad.stderr


def test_direct_input_output_graph(model_factory, tmp_path):
    path = model_factory([], {"x": (2, 3)}, {"x": (2, 3)})
    run = compile_model(path)
    assert run.metrics["cycles"] == 0
    assert run.verification["passed"]
    assert "no work" in render_report([("alias", run)], tmp_path / "r.html").read_text()
