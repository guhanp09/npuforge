#!/bin/sh
# Run inside a disposable Debian Python container with the repository at /source (read-only).
set -eu
apt-get update -qq
apt-get install -y --no-install-recommends build-essential git >/dev/null
mkdir /tmp/npuforge
tar -C /source --exclude='.venv*' --exclude=.git --exclude=output --exclude=build \
    --exclude=dist --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache \
    -cf - . | tar -C /tmp/npuforge -xf -
cd /tmp/npuforge
python -m pip install --quiet '.[dev]'
ruff check .
ruff format --check .
clang-format --dry-run --Werror cpp/include/*.hpp cpp/src/*.cpp cpp/tests/*.cpp
python -m pytest -q
cmake -S . -B build/native -DNPUFORGE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
npuforge demo
python examples/generate_models.py --output output/models
npuforge compile output/models/demo_gemm.onnx --compare --output output/gemm
python examples/compare_gemm_fusion.py
python -m build
python - <<'PY'
import json
from pathlib import Path

actual = json.loads(Path('output/results.json').read_text())
reference = json.loads(Path('benchmarks/results.json').read_text())
for a, b in zip(actual, reference, strict=True):
    assert a['case'] == b['case']
    assert a['metrics'] == b['metrics'], a['case']
    assert a['verification']['passed']
print('LINUX VERIFIED: all six native metric sets match the macOS reference exactly.')
actual = json.loads(Path('output/gemm-fusion/results.json').read_text())
reference = json.loads(Path('benchmarks/gemm_fusion_results.json').read_text())
for a, b in zip(actual, reference, strict=True):
    assert a['case'] == b['case']
    assert a['metrics'] == b['metrics'], a['case']
    assert a['verification']['passed']
print('LINUX VERIFIED: all four Gemm/Relu fusion metric sets match exactly.')
PY
