#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

MODE=${1:-smoke}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: bash scripts/15_verify_4090.sh [smoke|full]" >&2
  exit 2
fi

source scripts/python_env.sh

CONFIGURED_PYTHON=${PYTHON_BIN:-}
if ! PYTHON_BIN=$(find_python_bin cuda-torch); then
  fallback_python="none"
  if [[ -n "$CONFIGURED_PYTHON" ]]; then
    fallback_python=$CONFIGURED_PYTHON
  else
    unset PYTHON_BIN
    fallback_python=$(find_python_bin python3 2>/dev/null || true)
    [[ -n "$fallback_python" ]] || fallback_python="none"
  fi
  show_python_candidates >&2
  show_cuda_torch_remediation "$fallback_python"
  exit 1
fi
export PYTHON_BIN

echo "Using Python: $PYTHON_BIN"

chmod +x scripts/*.sh scripts/*.py debug_labs/*.py \
  vllm_learning/scripts/*.sh vllm_learning/examples/*.py

echo "[1/5] Handbook/script consistency"
"$PYTHON_BIN" scripts/verify_workspace.py --allow-generated

echo "[2/5] RTX 4090 and CUDA toolchain"
bash scripts/00_check_env.sh --strict

echo "[3/5] Clean sm_89 extension build"
bash scripts/clean_build.sh
bash scripts/10_build.sh

echo "[4/5] Extension, 24 exports and sm_89 cubin"
mkdir -p reports/preflight
"$PYTHON_BIN" debug_labs/preflight.py --json reports/preflight/4090_preflight.json

echo "[5/5] GPU tests ($MODE)"
if [[ "$MODE" == "full" ]]; then
  bash scripts/20_test.sh
else
  "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py \
    -k "all_expected_exports_exist or all_exports_use_current_stream"
fi

echo "RTX 4090 $MODE verification: PASS"
echo "Report: reports/preflight/4090_preflight.json"
