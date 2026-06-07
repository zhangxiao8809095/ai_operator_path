#!/usr/bin/env bash
set -euo pipefail
OP=${1:-gemm_tiled}
mkdir -p reports/ncu
ncu \
  --target-processes all \
  --set full \
  --kernel-name-base demangled \
  --export "reports/ncu/${OP}" \
  --force-overwrite \
  python benchmark/profile_entry.py "${OP}" --repeat 20
