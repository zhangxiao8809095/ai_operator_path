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
  local name candidate
  for name in python python3; do
    candidate=$(command -v "$name" 2>/dev/null || true)
    if [[ -x "$candidate" ]] && "$candidate" -c \
      'import sys; raise SystemExit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

FAILURES=0
run_required() {
  local label=$1
  shift
  printf '\n[%s]\n' "$label"
  if ! "$@"; then
    FAILURES=$((FAILURES + 1))
  fi
}

find_cuda_tool() {
  local name=$1
  local override=${2:-}
  if [[ -n "$override" && -x "$override" ]]; then
    printf '%s\n' "$override"
    return
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return
  fi
  local candidate
  if [[ -n ${CUDA_HOME:-} && -x "$CUDA_HOME/bin/$name" ]]; then
    printf '%s\n' "$CUDA_HOME/bin/$name"
    return
  fi
  for candidate in \
    "/usr/local/cuda/bin/$name" \
    "/usr/local/cuda-12.6/bin/$name" \
    "/usr/local/cuda-12.4/bin/$name" \
    "/usr/local/cuda-12.1/bin/$name"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

show_tool() {
  local name=$1
  local override=${2:-}
  local executable
  printf '\n[%s]\n' "$name"
  if executable=$(find_cuda_tool "$name" "$override"); then
    "$executable" --version || true
  else
    echo "NOT FOUND"
    if (( STRICT )); then
      FAILURES=$((FAILURES + 1))
    fi
  fi
}

run_required "1/7 GPU" nvidia-smi
if NVCC=$(find_cuda_tool nvcc); then
  run_required "2/7 CUDA compiler" "$NVCC" --version
else
  printf '\n[2/7 CUDA compiler]\nNOT FOUND\n'
  FAILURES=$((FAILURES + 1))
fi
show_tool ncu "${NCU_BIN:-}"
show_tool nsys "${NSYS_BIN:-}"
show_tool compute-sanitizer "${COMPUTE_SANITIZER_BIN:-}"
show_tool cuobjdump "${CUOBJDUMP_BIN:-}"

printf '\n[7/7 Python / PyTorch]\n'
if ! PYTHON_BIN=$(find_python_bin); then
  echo "Python 3 NOT FOUND"
  FAILURES=$((FAILURES + 1))
elif ! "$PYTHON_BIN" - <<'PY'
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
then
  FAILURES=$((FAILURES + 1))
fi

if (( FAILURES > 0 )); then
  if (( STRICT )); then
    echo "Environment check: FAIL ($FAILURES required check(s) failed)" >&2
    exit 1
  fi
  echo "Environment check: INCOMPLETE ($FAILURES item(s) unavailable; rerun with --strict after activating the server environment)"
else
  echo "Environment check: PASS"
fi
