#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

find_python_bin() {
  if [[ -n ${PYTHON_BIN:-} ]]; then
    local configured
    configured=$(command -v "$PYTHON_BIN" 2>/dev/null || true)
    if [[ -z "$configured" && -x "$PYTHON_BIN" ]]; then
      configured=$PYTHON_BIN
    fi
    [[ -n "$configured" ]] || return 1
    printf '%s\n' "$configured"
    return
  fi
  command -v python 2>/dev/null || command -v python3 2>/dev/null
}

if ! PYTHON_BIN=$(find_python_bin); then
  echo "error: Python 3 was not found; activate the intended environment or set PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import torch; assert torch.version.cuda is not None' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: the active Python environment does not contain a CUDA-enabled PyTorch.

On a rental image that already provides CUDA PyTorch, recreate the environment with:
  python3 -m venv --system-site-packages .venv

Otherwise install the CUDA PyTorch wheel selected for that server before running this script.
Keep the operator project and vllm_learning in separate virtual environments.
EOF
  exit 1
fi

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=${MAX_JOBS:-2}
"$PYTHON_BIN" -m pip install -U pip setuptools wheel ninja pytest
# 开发阶段建议不使用 build isolation，减少 PyTorch/CUDA extension 构建问题。
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import aiop4090
import aiop4090._C as extension

print("aiop4090:", Path(aiop4090.__file__).resolve())
print("extension:", Path(extension.__file__).resolve())
PY
