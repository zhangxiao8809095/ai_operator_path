#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

STRICT=0
if [[ ${1:-} == "--strict" ]]; then
  STRICT=1
elif (( $# > 0 )); then
  echo "usage: bash scripts/00_check_env.sh [--strict]" >&2
  exit 2
fi

source scripts/python_env.sh

find_cuda_tool() {
  local name=$1
  local override=${2:-}
  if [[ -n "$override" && -x "$override" ]]; then printf '%s\n' "$override"; return; fi
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
  if [[ -n ${CUDA_HOME:-} && -x "$CUDA_HOME/bin/$name" ]]; then
    printf '%s\n' "$CUDA_HOME/bin/$name"
    return
  fi
  local candidate
  for candidate in \
    "/usr/local/cuda/bin/$name" \
    "/usr/local/cuda-12.6/bin/$name" \
    "/usr/local/cuda-12.4/bin/$name" \
    "/usr/local/cuda-12.1/bin/$name"; do
    if [[ -x "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  done
  return 1
}

FAILURES=0
show_required_command() {
  local label=$1
  shift
  printf '\n[%s]\n' "$label"
  if ! "$@"; then FAILURES=$((FAILURES + 1)); fi
}

show_cuda_tool() {
  local name=$1
  local override=${2:-}
  local executable
  printf '\n[%s]\n' "$name"
  if executable=$(find_cuda_tool "$name" "$override"); then
    "$executable" --version || true
  else
    echo "NOT FOUND"
    if (( STRICT )); then FAILURES=$((FAILURES + 1)); fi
  fi
}

show_required_command "1/7 GPU" nvidia-smi
if NVCC=$(find_cuda_tool nvcc); then
  show_required_command "2/7 CUDA compiler" "$NVCC" --version
else
  printf '\n[2/7 CUDA compiler]\nNOT FOUND\n'
  FAILURES=$((FAILURES + 1))
fi
show_cuda_tool ncu "${NCU_BIN:-}"
show_cuda_tool nsys "${NSYS_BIN:-}"
show_cuda_tool compute-sanitizer "${COMPUTE_SANITIZER_BIN:-}"
show_cuda_tool cuobjdump "${CUOBJDUMP_BIN:-}"

printf '\n[7/7 Python / PyTorch]\n'
CONFIGURED_PYTHON=${PYTHON_BIN:-}
if ! PYTHON_BIN=$(find_python_bin cuda-torch); then
  if [[ -n "$CONFIGURED_PYTHON" ]]; then
    PYTHON_BIN=$CONFIGURED_PYTHON
  else
    unset PYTHON_BIN
    PYTHON_BIN=$(find_python_bin python3 2>/dev/null || true)
  fi
fi

if [[ -z ${PYTHON_BIN:-} ]]; then
  echo "Python 3 NOT FOUND"
  FAILURES=$((FAILURES + 1))
else
  PYTHON_STATUS=0
  "$PYTHON_BIN" - <<'PY' || PYTHON_STATUS=$?
import sys
print("python:", sys.executable)
try:
    import torch
except Exception as error:
    print("torch import: FAIL", repr(error))
    raise SystemExit(1)
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(2)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
if torch.cuda.get_device_capability(0) != (8, 9):
    raise SystemExit(3)
PY
  if (( PYTHON_STATUS != 0 )); then
    if (( PYTHON_STATUS == 1 )); then
      show_python_candidates >&2
      show_cuda_torch_remediation "$PYTHON_BIN"
    elif (( PYTHON_STATUS == 2 )); then
      echo "error: PyTorch has CUDA support, but torch.cuda.is_available() is false." >&2
      echo "Check nvidia-smi, driver/container GPU access, and CUDA_VISIBLE_DEVICES." >&2
    elif (( PYTHON_STATUS == 3 )); then
      echo "error: CUDA is available, but device 0 is not an RTX 4090 (compute capability 8.9)." >&2
    fi
    FAILURES=$((FAILURES + 1))
  fi
fi

if (( FAILURES > 0 )); then
  if (( STRICT )); then
    echo "Environment check: FAIL ($FAILURES required check(s) failed)" >&2
    exit 1
  fi
  echo "Environment check: INCOMPLETE ($FAILURES item(s) unavailable; rerun with --strict in the server environment)"
else
  echo "Environment check: PASS"
fi
