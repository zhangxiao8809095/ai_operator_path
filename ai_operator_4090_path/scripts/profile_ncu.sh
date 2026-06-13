#!/usr/bin/env bash
set -euo pipefail
OP=${1:-gemm_tiled}
mkdir -p reports/ncu
# 第一次建议用 speed-of-light；后面再加 MemoryWorkloadAnalysis / SchedulerStats。
ncu \
  --target-processes all \
  --set speed-of-light \
  --force-overwrite \
  -o reports/ncu/${OP}_sol \
  python benchmark/profile_entry.py --op ${OP} --iters 20

echo "Generated: reports/ncu/${OP}_sol.ncu-rep"
