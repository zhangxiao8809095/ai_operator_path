#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

EXPERIMENT=${1:-all}

find_python_bin() {
  if [[ -n ${PYTHON_BIN:-} ]]; then
    local configured
    configured=$(command -v "$PYTHON_BIN" 2>/dev/null || true)
    if [[ -z "$configured" && -x "$PYTHON_BIN" ]]; then configured=$PYTHON_BIN; fi
    [[ -n "$configured" ]] || return 1
    printf '%s\n' "$configured"
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

mkdir -p reports/gemm reports/benchmark reports/ncu reports/nsys

run_log() {
  local log_file=$1
  shift
  "$@" 2>&1 | tee "$log_file"
}

append_log() {
  local log_file=$1
  shift
  "$@" 2>&1 | tee -a "$log_file"
}

extract_ncu_evidence() {
  local output_file=$1
  local report_file=$2
  local kernel_candidates=$3
  local temp_dir kernel_name
  local -a candidates
  temp_dir=$(mktemp -d)
  IFS='|' read -r -a candidates <<<"$kernel_candidates"
  for kernel_name in "${candidates[@]}"; do
    if bash scripts/extract_ncu_metrics.sh "$report_file" "$kernel_name" 6 \
        >"$temp_dir/fixed" 2>"$temp_dir/error" && \
       bash scripts/extract_ncu_supplemental_metrics.sh "$report_file" "$kernel_name" 6 \
        >"$temp_dir/supplemental" 2>>"$temp_dir/error"; then
      {
        printf '# Fixed eight metrics\n\n'
        cat "$temp_dir/fixed"
        printf '\n# GEMM supplemental metrics\n\n'
        cat "$temp_dir/supplemental"
      } >"$output_file"
      rm -rf "$temp_dir"
      return
    fi
  done
  cat "$temp_dir/error" >&2
  rm -rf "$temp_dir"
  echo "error: no NCU kernel candidate matched $report_file: $kernel_candidates" >&2
  return 1
}

run_c01() {
  run_log reports/gemm/GEMM-C01.log \
    "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py -k gemm_boundary_matrix
}

run_c02() {
  run_log reports/gemm/GEMM-C02.log \
    "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py -k gemm_empty_dimensions
}

run_c03() {
  run_log reports/gemm/GEMM-C03.log \
    "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py -k gemm_wmma_error_over_k
  append_log reports/gemm/GEMM-C03.log \
    "$PYTHON_BIN" benchmark/gemm_validation.py --output reports/gemm/GEMM-C03.csv
}

run_d01() {
  local log=reports/gemm/GEMM-D01.log
  run_log "$log" bash scripts/40_sanitize.sh memcheck gemm
  append_log "$log" bash scripts/40_sanitize.sh initcheck gemm
  append_log "$log" bash scripts/40_sanitize.sh synccheck gemm
}

run_p01() {
  local log=reports/gemm/GEMM-P01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -s -k gemm_boundary_matrix
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py \
    --op gemm --csv reports/benchmark/gemm.csv
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_naive
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_tiled
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_tiled_padding
  extract_ncu_evidence reports/gemm/GEMM-P01-naive.md \
    reports/ncu/gemm_naive_full.ncu-rep gemm_naive_kernel
  extract_ncu_evidence reports/gemm/GEMM-P01-tiled.md \
    reports/ncu/gemm_tiled_full.ncu-rep gemm_tiled_kernel
  extract_ncu_evidence reports/gemm/GEMM-P01-tiled-padding.md \
    reports/ncu/gemm_tiled_padding_full.ncu-rep gemm_tiled_padding_kernel
}

run_p02() {
  local log=reports/gemm/GEMM-P02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -s -k gemm_boundary_matrix
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py \
    --op gemm --csv reports/benchmark/gemm.csv
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_tiled
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_regtile2x2
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_regtile4x4
  extract_ncu_evidence reports/gemm/GEMM-P02-tiled.md \
    reports/ncu/gemm_tiled_full.ncu-rep gemm_tiled_kernel
  extract_ncu_evidence reports/gemm/GEMM-P02-regtile2x2.md \
    reports/ncu/gemm_regtile2x2_full.ncu-rep gemm_regtile2x2_kernel
  extract_ncu_evidence reports/gemm/GEMM-P02-regtile4x4.md \
    reports/ncu/gemm_regtile4x4_full.ncu-rep gemm_regtile4x4_kernel
}

run_p03() {
  local log=reports/gemm/GEMM-P03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -s -k float4_misaligned
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_vectorized_float4
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_vectorized_float4_misaligned
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_vectorized_float4_tail
  extract_ncu_evidence reports/gemm/GEMM-P03-aligned.md \
    reports/ncu/gemm_vectorized_float4_full.ncu-rep gemm_vectorized_float4_kernel
  extract_ncu_evidence reports/gemm/GEMM-P03-misaligned.md \
    reports/ncu/gemm_vectorized_float4_misaligned_full.ncu-rep gemm_vectorized_float4_kernel
  extract_ncu_evidence reports/gemm/GEMM-P03-tail.md \
    reports/ncu/gemm_vectorized_float4_tail_full.ncu-rep gemm_vectorized_float4_kernel
}

run_p04() {
  local log=reports/gemm/GEMM-P04.log
  run_log "$log" "$PYTHON_BIN" benchmark/gemm_validation.py \
    --output reports/gemm/GEMM-C03.csv
  append_log "$log" bash scripts/profile_ncu_full.sh gemm_wmma_fp16
  extract_ncu_evidence reports/gemm/GEMM-P04-wmma.md \
    reports/ncu/gemm_wmma_fp16_full.ncu-rep gemm_wmma_fp16_kernel
  append_log "$log" bash scripts/profile_nsys.sh gemm_wmma_fp16
  append_log "$log" bash scripts/profile_nsys.sh gemm_wmma_from_fp32
  append_log "$log" bash scripts/profile_nsys.sh gemm_wmma_fallback
}

case "$EXPERIMENT" in
  GEMM-C01|gemm-c01) run_c01 ;;
  GEMM-C02|gemm-c02) run_c02 ;;
  GEMM-C03|gemm-c03) run_c03 ;;
  GEMM-D01|gemm-d01) run_d01 ;;
  GEMM-P01|gemm-p01) run_p01 ;;
  GEMM-P02|gemm-p02) run_p02 ;;
  GEMM-P03|gemm-p03) run_p03 ;;
  GEMM-P04|gemm-p04) run_p04 ;;
  all)
    for experiment in \
      GEMM-C01 GEMM-C02 GEMM-C03 GEMM-D01 \
      GEMM-P01 GEMM-P02 GEMM-P03 GEMM-P04; do
      "$SCRIPT_DIR/run_gemm_experiment.sh" "$experiment"
    done
    ;;
  *)
    echo "unknown GEMM experiment: $EXPERIMENT" >&2
    echo "choose GEMM-C01/C02/C03/D01/P01/P02/P03/P04 or all" >&2
    exit 2
    ;;
esac
