#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

source scripts/python_env.sh

if ! PYTHON_BIN=$(find_python_bin cuda-torch); then
  fallback_python=$(find_python_bin python3 2>/dev/null || true)
  show_python_candidates >&2
  show_cuda_torch_remediation "${fallback_python:-none}"
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import torch; assert torch.version.cuda is not None' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: the active Python environment does not contain a CUDA-enabled PyTorch.

If the rental image already provides CUDA PyTorch, create the environment with:
  python3 -m venv --system-site-packages .venv

Otherwise install the CUDA PyTorch wheel selected for that server first.
Keep this extension project and vllm_learning in separate virtual environments.
EOF
  exit 1
fi

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=${MAX_JOBS:-2}
"$PYTHON_BIN" -m pip install -U pip setuptools wheel ninja pytest
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import aiop4090
import aiop4090._C as extension
print("aiop4090:", Path(aiop4090.__file__).resolve())
print("extension:", Path(extension.__file__).resolve())
PY
