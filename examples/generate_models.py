"""Run after installation: python examples/generate_models.py --output output/models."""

import argparse
from pathlib import Path

from npuforge.compiler import optimize
from npuforge.importer import import_onnx
from npuforge.models import generate_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/models"))
args = parser.parse_args()
for kind in ("matmul", "mlp", "gemm", "gemm_relu"):
    path = generate_model(args.output / f"demo_{kind}.onnx", kind)
    original = import_onnx(path)
    optimized, changes = optimize(original)
    print(f"\n{path}\n\nOriginal:\n{original}\n\nOptimized:\n{optimized}\n{changes}")
