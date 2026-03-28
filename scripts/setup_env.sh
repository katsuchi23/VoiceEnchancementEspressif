#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTORCH_FLAVOR="${PYTORCH_FLAVOR:-cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
PYTORCH_CUDA_FLAVOR="${PYTORCH_CUDA_FLAVOR:-cu124}"
cd "$REPO_ROOT"

case "$PYTORCH_FLAVOR" in
  cpu)
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    ;;
  gpu)
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${PYTORCH_CUDA_FLAVOR}}"
    ;;
  *)
    echo "Unsupported PYTORCH_FLAVOR='$PYTORCH_FLAVOR'. Use 'cpu' or 'gpu'."
    exit 1
    ;;
esac

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [[ "$MODE" == "base" ]]; then
  pip install -r requirements/base.txt
else
  pip install -r requirements/dev.txt
fi

pip install \
  --index-url "$PYTORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}" \
  "torchaudio==${TORCH_VERSION}"

echo "Environment setup complete (mode=$MODE, pytorch_flavor=$PYTORCH_FLAVOR, index=$PYTORCH_INDEX_URL)."
