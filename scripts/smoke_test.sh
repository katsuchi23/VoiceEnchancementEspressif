#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Run: make setup"
  exit 1
fi

.venv/bin/python - <<'PY'
import importlib
modules = [
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "pandas",
    "yaml",
    "requests",
    "remotezip",
    "einops",
    "torch",
    "torchaudio",
    "onnx",
    "onnxruntime",
    "pesq",
    "pystoi",
]
for name in modules:
    importlib.import_module(name)
print("Smoke test passed: all imports succeeded.")
PY
