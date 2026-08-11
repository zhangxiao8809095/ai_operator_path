#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

OP=${1:-all}
ITERS=${ITERS:-30}
mkdir -p reports/nsys

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

find_nsys() {
  if [[ -n ${NSYS_BIN:-} && -x ${NSYS_BIN:-} ]]; then
    printf '%s\n' "$NSYS_BIN"
    return
  fi
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/nsys \
    /usr/local/cuda-12.6/bin/nsys \
    /usr/local/cuda-12.4/bin/nsys \
    /usr/local/cuda-12.1/bin/nsys; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if ! NSYS=$(find_nsys); then
  echo "error: nsys was not found; add it to PATH or set NSYS_BIN" >&2
  exit 1
fi

if [[ "$OP" == "all" ]]; then
  TARGET=("$PYTHON_BIN" benchmark/bench_ops.py --op all)
  OUTPUT="reports/nsys/bench_ops_all"
else
  TARGET=("$PYTHON_BIN" benchmark/profile_entry.py --op "$OP" --iters "$ITERS")
  OUTPUT="reports/nsys/${OP}"
fi

"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o "$OUTPUT" \
  "${TARGET[@]}"

echo "Generated: ${OUTPUT}.nsys-rep"
