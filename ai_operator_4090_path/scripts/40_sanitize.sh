#!/usr/bin/env bash
set -euo pipefail

TOOL=${1:-memcheck}
GROUP=${2:-all}

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

case "$TOOL" in
  memcheck|racecheck|initcheck|synccheck) ;;
  *)
    echo "Unsupported tool: $TOOL" >&2
    echo "Choose: memcheck, racecheck, initcheck, synccheck" >&2
    exit 2
    ;;
esac

case "$GROUP" in
  gemm)
    TEST_ARGS=(tests/test_gemm.py tests/test_operator_validation.py -k gemm)
    ;;
  softmax)
    TEST_ARGS=(tests/test_softmax_norm.py tests/test_operator_validation.py -k softmax)
    ;;
  norm)
    TEST_ARGS=(tests/test_softmax_norm.py tests/test_operator_validation.py -k "layernorm or rmsnorm")
    ;;
  attention)
    TEST_ARGS=(tests/test_attention.py tests/test_operator_validation.py -k attention)
    ;;
  all)
    TEST_ARGS=(tests)
    ;;
  *)
    echo "Unsupported group: $GROUP" >&2
    echo "Choose: gemm, softmax, norm, attention, all" >&2
    exit 2
    ;;
esac

find_compute_sanitizer() {
  if [[ -n ${COMPUTE_SANITIZER_BIN:-} && -x ${COMPUTE_SANITIZER_BIN:-} ]]; then
    printf '%s\n' "$COMPUTE_SANITIZER_BIN"
    return
  fi
  if command -v compute-sanitizer >/dev/null 2>&1; then
    command -v compute-sanitizer
    return
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/compute-sanitizer \
    /usr/local/cuda-12.6/bin/compute-sanitizer \
    /usr/local/cuda-12.4/bin/compute-sanitizer \
    /usr/local/cuda-12.1/bin/compute-sanitizer; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if ! COMPUTE_SANITIZER=$(find_compute_sanitizer); then
  echo "compute-sanitizer was not found; add it to PATH or set COMPUTE_SANITIZER_BIN" >&2
  exit 1
fi

echo "Running Compute Sanitizer tool=$TOOL group=$GROUP"
"$COMPUTE_SANITIZER" --tool "$TOOL" \
  "$PYTHON_BIN" -m pytest -q "${TEST_ARGS[@]}"
