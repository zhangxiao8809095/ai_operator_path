#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

EXPERIMENT=${1:-all}

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

mkdir -p reports/{softmax,layernorm,rmsnorm,attention,benchmark,ncu,nsys}

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
        printf '\n# Supplemental metrics\n\n'
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

collect_ncu() {
  local log_file=$1
  local evidence_file=$2
  local profile_op=$3
  local kernel_name=$4
  append_log "$log_file" bash scripts/profile_ncu_full.sh "$profile_op"
  extract_ncu_evidence "$evidence_file" \
    "reports/ncu/${profile_op}_full.ncu-rep" "$kernel_name"
}

run_softmax_c01() {
  run_log reports/softmax/SM-C01.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k softmax_shape_properties
}

run_softmax_c02() {
  run_log reports/softmax/SM-C02.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k "softmax_shape_properties or softmax_extreme"
}

run_softmax_d01() {
  local log=reports/softmax/SM-D01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py \
    -k softmax_repeated_execution_is_stable
  append_log "$log" bash scripts/40_sanitize.sh racecheck softmax
  append_log "$log" bash scripts/40_sanitize.sh synccheck softmax
}

run_softmax_p01() {
  local log=reports/softmax/SM-P01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k softmax_shape_properties
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op softmax \
    --csv reports/benchmark/softmax.csv
  collect_ncu "$log" reports/softmax/SM-P01-block.md \
    softmax_block_reduce softmax_block_reduce_kernel
  collect_ncu "$log" reports/softmax/SM-P01-warp.md \
    softmax_warp_reduce softmax_warp_reduce_kernel
}

run_softmax_p02() {
  local log=reports/softmax/SM-P02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k softmax_shape_properties
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op softmax \
    --csv reports/benchmark/softmax.csv
  collect_ncu "$log" reports/softmax/SM-P02-block.md \
    softmax_block_reduce softmax_block_reduce_kernel
  collect_ncu "$log" reports/softmax/SM-P02-warp.md \
    softmax_warp_reduce softmax_warp_reduce_kernel
  collect_ncu "$log" reports/softmax/SM-P02-online.md \
    softmax_online softmax_online_kernel
}

run_softmax_p03() {
  local log=reports/softmax/SM-P03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k softmax_shape_properties
  for op in softmax_row_small softmax_block_reduce_small softmax_warp_reduce_small softmax_online_small; do
    append_log "$log" bash scripts/profile_nsys.sh "$op"
  done
}

run_layernorm_c01() {
  run_log reports/layernorm/LN-C01.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_softmax_norm.py tests/test_operator_validation.py -k layernorm
}

run_layernorm_c02() {
  local log=reports/layernorm/LN-C02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k "layernorm_numerical_inputs or layernorm_boundary_and_eps"
  append_log "$log" "$PYTHON_BIN" benchmark/operator_validation_evidence.py \
    --experiment layernorm-numerics --output reports/layernorm/LN-C02.csv
}

run_layernorm_c03() {
  run_log reports/layernorm/LN-C03.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k "norm_vectorized_misaligned_fallbacks or layernorm_boundary_and_eps"
}

run_layernorm_p01() {
  local log=reports/layernorm/LN-P01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k layernorm_boundary_and_eps
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op layernorm \
    --csv reports/benchmark/layernorm.csv
  collect_ncu "$log" reports/layernorm/LN-P01-row.md layernorm_row layernorm_row_kernel
  collect_ncu "$log" reports/layernorm/LN-P01-block.md \
    layernorm_block_reduce layernorm_block_reduce_kernel
}

run_layernorm_p02() {
  local log=reports/layernorm/LN-P02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k layernorm_boundary_and_eps
  collect_ncu "$log" reports/layernorm/LN-P02-block.md \
    layernorm_block_reduce layernorm_block_reduce_kernel
  collect_ncu "$log" reports/layernorm/LN-P02-warp.md \
    layernorm_warp_reduce layernorm_warp_reduce_kernel
}

run_layernorm_p03() {
  local log=reports/layernorm/LN-P03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py \
    -k norm_vectorized_misaligned_fallbacks
  collect_ncu "$log" reports/layernorm/LN-P03-aligned.md \
    layernorm_vectorized layernorm_vectorized_kernel
  collect_ncu "$log" reports/layernorm/LN-P03-misaligned.md \
    layernorm_vectorized_misaligned layernorm_warp_reduce_kernel
  collect_ncu "$log" reports/layernorm/LN-P03-tail.md \
    layernorm_vectorized_tail layernorm_warp_reduce_kernel
  append_log "$log" bash scripts/profile_nsys.sh layernorm_vectorized
  append_log "$log" bash scripts/profile_nsys.sh layernorm_vectorized_misaligned
  append_log "$log" bash scripts/profile_nsys.sh layernorm_vectorized_tail
}

run_rmsnorm_c01() {
  run_log reports/rmsnorm/RMS-C01.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_softmax_norm.py tests/test_operator_validation.py -k rmsnorm
}

run_rmsnorm_c02() {
  local log=reports/rmsnorm/RMS-C02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k rmsnorm_numerical_inputs
  append_log "$log" "$PYTHON_BIN" benchmark/operator_validation_evidence.py \
    --experiment rmsnorm-numerics --output reports/rmsnorm/RMS-C02.csv
}

run_rmsnorm_c03() {
  run_log reports/rmsnorm/RMS-C03.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k "norm_vectorized_misaligned_fallbacks or rmsnorm_boundary_and_eps"
}

run_rmsnorm_p01() {
  local log=reports/rmsnorm/RMS-P01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k rmsnorm_boundary_and_eps
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op rmsnorm \
    --csv reports/benchmark/rmsnorm.csv
  collect_ncu "$log" reports/rmsnorm/RMS-P01-row.md rmsnorm_row rmsnorm_row_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P01-block.md \
    rmsnorm_block_reduce rmsnorm_block_reduce_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P01-warp.md \
    rmsnorm_warp_reduce rmsnorm_warp_reduce_kernel
}

run_rmsnorm_p02() {
  local log=reports/rmsnorm/RMS-P02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s tests/test_operator_validation.py \
    -k norm_vectorized_misaligned_fallbacks
  collect_ncu "$log" reports/rmsnorm/RMS-P02-float2.md \
    rmsnorm_vectorized rmsnorm_vectorized_float2_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P02-float4.md \
    rmsnorm_vectorized_float4 rmsnorm_vectorized_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P02-float2-only.md \
    rmsnorm_vectorized_float2_only rmsnorm_vectorized_float2_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P02-float4-fallback.md \
    rmsnorm_vectorized_float4_fallback rmsnorm_warp_reduce_kernel
  append_log "$log" bash scripts/profile_nsys.sh rmsnorm_vectorized_float4_misaligned
}

run_rmsnorm_p03() {
  local log=reports/rmsnorm/RMS-P03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k "layernorm_boundary_and_eps or rmsnorm_boundary_and_eps"
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op norm \
    --csv reports/benchmark/norm.csv
  collect_ncu "$log" reports/rmsnorm/RMS-P03-layernorm-warp.md \
    layernorm_warp_reduce layernorm_warp_reduce_kernel
  collect_ncu "$log" reports/rmsnorm/RMS-P03-rmsnorm-warp.md \
    rmsnorm_warp_reduce rmsnorm_warp_reduce_kernel
}

run_attention_c01() {
  run_log reports/attention/AT-C01.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py -k attention_boundary_shapes
}

run_attention_c02() {
  run_log reports/attention/AT-C02.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_attention.py tests/test_operator_validation.py -k causal
}

run_attention_c03() {
  local log=reports/attention/AT-C03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py \
    -k "attention_kv_cache_boundaries or attention_rejects_invalid_kv_len"
  append_log "$log" "$PYTHON_BIN" benchmark/operator_validation_evidence.py \
    --experiment attention-kv --output reports/attention/AT-C03.csv
}

run_attention_c04() {
  run_log reports/attention/AT-C04.log "$PYTHON_BIN" -m pytest -q -s \
    tests/test_operator_validation.py \
    -k "attention_empty_batch_and_invalid_reduction_dims or attention_rejects_invalid_contracts"
}

run_attention_p01() {
  local log=reports/attention/AT-P01.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k attention_boundary_shapes
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op attention \
    --csv reports/benchmark/attention.csv
  collect_ncu "$log" reports/attention/AT-P01-naive-causal.md \
    attention_naive 'attention_naive_kernel<true>|attention_naive_kernel<(bool)1>'
  collect_ncu "$log" reports/attention/AT-P01-naive-noncausal.md \
    attention_naive_noncausal 'attention_naive_kernel<false>|attention_naive_kernel<(bool)0>'
  collect_ncu "$log" reports/attention/AT-P01-online-causal.md \
    attention_tiled_online_softmax attention_tiled_online_softmax_kernel
  collect_ncu "$log" reports/attention/AT-P01-online-noncausal.md \
    attention_tiled_online_softmax_noncausal attention_tiled_online_softmax_kernel
}

run_attention_p02() {
  local log=reports/attention/AT-P02.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k attention_boundary_shapes
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op attention \
    --csv reports/benchmark/attention.csv
  collect_ncu "$log" reports/attention/AT-P02-naive-s64.md \
    attention_naive_s64 'attention_naive_kernel<true>|attention_naive_kernel<(bool)1>'
  collect_ncu "$log" reports/attention/AT-P02-online-s64.md \
    attention_tiled_online_s64 attention_tiled_online_softmax_kernel
  collect_ncu "$log" reports/attention/AT-P02-naive-s128.md \
    attention_naive 'attention_naive_kernel<true>|attention_naive_kernel<(bool)1>'
  collect_ncu "$log" reports/attention/AT-P02-online-s128.md \
    attention_tiled_online_softmax attention_tiled_online_softmax_kernel
  append_log "$log" bash scripts/profile_nsys.sh attention_tiled_online_softmax
}

run_attention_p03() {
  local log=reports/attention/AT-P03.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py \
    -k attention_kv_cache_boundaries
  append_log "$log" "$PYTHON_BIN" benchmark/bench_ops.py --op attention \
    --csv reports/benchmark/attention.csv
  for op in attention_kv_cache_decode_kv1 attention_kv_cache_decode_kv32 \
    attention_kv_cache_decode_kv128 attention_kv_cache_decode_kv256; do
    append_log "$log" bash scripts/profile_nsys.sh "$op"
  done
  collect_ncu "$log" reports/attention/AT-P03-kv256.md \
    attention_kv_cache_decode_kv256 attention_kv_cache_decode_kernel
}

run_attention_p04() {
  local log=reports/attention/AT-P04.log
  run_log "$log" "$PYTHON_BIN" -m pytest -q tests/test_operator_validation.py -k causal
  collect_ncu "$log" reports/attention/AT-P04-naive-causal.md \
    attention_naive 'attention_naive_kernel<true>|attention_naive_kernel<(bool)1>'
  collect_ncu "$log" reports/attention/AT-P04-naive-noncausal.md \
    attention_naive_noncausal 'attention_naive_kernel<false>|attention_naive_kernel<(bool)0>'
  collect_ncu "$log" reports/attention/AT-P04-fixed-causal.md \
    attention_causal_naive 'attention_naive_kernel<true>|attention_naive_kernel<(bool)1>'
  append_log "$log" bash scripts/profile_nsys.sh attention_naive
  append_log "$log" bash scripts/profile_nsys.sh attention_naive_noncausal
  append_log "$log" bash scripts/profile_nsys.sh attention_causal_naive
}

run_named() {
  case "$1" in
    SM-C01) run_softmax_c01 ;; SM-C02) run_softmax_c02 ;; SM-D01) run_softmax_d01 ;;
    SM-P01) run_softmax_p01 ;; SM-P02) run_softmax_p02 ;; SM-P03) run_softmax_p03 ;;
    LN-C01) run_layernorm_c01 ;; LN-C02) run_layernorm_c02 ;; LN-C03) run_layernorm_c03 ;;
    LN-P01) run_layernorm_p01 ;; LN-P02) run_layernorm_p02 ;; LN-P03) run_layernorm_p03 ;;
    RMS-C01) run_rmsnorm_c01 ;; RMS-C02) run_rmsnorm_c02 ;; RMS-C03) run_rmsnorm_c03 ;;
    RMS-P01) run_rmsnorm_p01 ;; RMS-P02) run_rmsnorm_p02 ;; RMS-P03) run_rmsnorm_p03 ;;
    AT-C01) run_attention_c01 ;; AT-C02) run_attention_c02 ;; AT-C03) run_attention_c03 ;;
    AT-C04) run_attention_c04 ;; AT-P01) run_attention_p01 ;; AT-P02) run_attention_p02 ;;
    AT-P03) run_attention_p03 ;; AT-P04) run_attention_p04 ;;
    *) return 2 ;;
  esac
}

run_group() {
  local experiment
  for experiment in "$@"; do
    "$SCRIPT_DIR/run_operator_experiment.sh" "$experiment"
  done
}

case "$EXPERIMENT" in
  softmax) run_group SM-C01 SM-C02 SM-D01 SM-P01 SM-P02 SM-P03 ;;
  layernorm) run_group LN-C01 LN-C02 LN-C03 LN-P01 LN-P02 LN-P03 ;;
  rmsnorm) run_group RMS-C01 RMS-C02 RMS-C03 RMS-P01 RMS-P02 RMS-P03 ;;
  attention) run_group AT-C01 AT-C02 AT-C03 AT-C04 AT-P01 AT-P02 AT-P03 AT-P04 ;;
  all)
    run_group \
      SM-C01 SM-C02 SM-D01 SM-P01 SM-P02 SM-P03 \
      LN-C01 LN-C02 LN-C03 LN-P01 LN-P02 LN-P03 \
      RMS-C01 RMS-C02 RMS-C03 RMS-P01 RMS-P02 RMS-P03 \
      AT-C01 AT-C02 AT-C03 AT-C04 AT-P01 AT-P02 AT-P03 AT-P04
    ;;
  SM-C01|SM-C02|SM-D01|SM-P01|SM-P02|SM-P03|\
  LN-C01|LN-C02|LN-C03|LN-P01|LN-P02|LN-P03|\
  RMS-C01|RMS-C02|RMS-C03|RMS-P01|RMS-P02|RMS-P03|\
  AT-C01|AT-C02|AT-C03|AT-C04|AT-P01|AT-P02|AT-P03|AT-P04)
    run_named "$EXPERIMENT"
    ;;
  *)
    echo "unknown operator experiment: $EXPERIMENT" >&2
    echo "choose an SM/LN/RMS/AT experiment id, family name, or all" >&2
    exit 2
    ;;
esac
