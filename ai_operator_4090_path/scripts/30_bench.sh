#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
if [[ -z ${PYTHON_BIN:-} ]]; then
  PYTHON_BIN=$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)
fi
[[ -n ${PYTHON_BIN:-} ]] || { echo "error: Python 3 was not found" >&2; exit 1; }
"$PYTHON_BIN" benchmark/bench_ops.py --op all "$@"
