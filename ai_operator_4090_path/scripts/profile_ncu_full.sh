#!/usr/bin/env bash
set -euo pipefail
OP=${1:-gemm_tiled}
mkdir -p reports/ncu
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --force-overwrite \
  -o reports/ncu/${OP}_full \
  python benchmark/profile_entry.py --op ${OP} --iters 10

echo "Generated: reports/ncu/${OP}_full.ncu-rep"
