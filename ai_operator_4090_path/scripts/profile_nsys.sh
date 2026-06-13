#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports/nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o reports/nsys/bench_ops \
  python benchmark/bench_ops.py --op all

echo "Generated: reports/nsys/bench_ops.nsys-rep"
