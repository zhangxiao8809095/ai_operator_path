#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports/nsys
nsys profile \
  -o reports/nsys/operator_lab \
  --force-overwrite true \
  --trace=cuda,nvtx,osrt \
  python benchmark/bench_ops.py
