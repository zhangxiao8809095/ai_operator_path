#!/usr/bin/env bash
set -euo pipefail

OP=${1:-gemm_naive}
if (( $# > 0 )); then
  shift
fi
PROFILE_ARGS=("$@")
ITERS=${ITERS:-20}
OUT_DIR=${OUT_DIR:-reports/ncu}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}

find_python_bin() {
  if [[ -n ${PYTHON_BIN:-} ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  local name candidate
  for name in python python3; do
    candidate=$(command -v "$name" 2>/dev/null || true)
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys; raise SystemExit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if ! PYTHON_BIN=$(find_python_bin); then
  echo "error: Python 3 was not found; set PYTHON_BIN explicitly" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

TORCH_LIB=$($PYTHON_BIN -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

find_ncu() {
  if [[ -n ${NCU_BIN:-} && -x ${NCU_BIN:-} ]]; then
    printf '%s\n' "$NCU_BIN"
    return
  fi
  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-12.6/bin/ncu \
    /usr/local/cuda-12.4/bin/ncu \
    /usr/local/cuda-12.1/bin/ncu; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if ! NCU=$(find_ncu); then
  echo "error: ncu was not found; add it to PATH or set NCU_BIN" >&2
  exit 1
fi

CUDA_ROOT=${CUDA_HOME:-$(cd "$(dirname "$NCU")/.." && pwd)}
NCU_COMMAND=(
  env
  PATH="$CUDA_ROOT/bin:$PATH"
  LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  TORCH_CUDA_ARCH_LIST="8.9"
  "$NCU"
  --target-processes all
  --set speed-of-light
  --force-overwrite
  -o "$OUT_DIR/${OP}_sol"
  "$PYTHON_BIN" benchmark/profile_entry.py
  --op "$OP"
  --iters "$ITERS"
  "${PROFILE_ARGS[@]}"
)

if [[ "$NCU_USE_SUDO" == "1" ]]; then
  command -v sudo >/dev/null 2>&1 || {
    echo "error: NCU_USE_SUDO=1 but sudo is unavailable" >&2
    exit 1
  }
  sudo "${NCU_COMMAND[@]}"
elif [[ "$NCU_USE_SUDO" == "0" ]]; then
  if ! "${NCU_COMMAND[@]}"; then
    echo "hint: if GPU performance counters are restricted, retry with NCU_USE_SUDO=1" >&2
    exit 1
  fi
else
  echo "error: NCU_USE_SUDO must be 0 or 1" >&2
  exit 2
fi
