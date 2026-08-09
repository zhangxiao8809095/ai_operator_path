#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  extract_ncu_supplemental_metrics.sh <report.ncu-rep> <kernel_name> [invocation]

Example:
  bash scripts/extract_ncu_supplemental_metrics.sh \
    reports/ncu/gemm_regtile2x2_full.ncu-rep \
    gemm_regtile2x2_kernel \
    6

Arguments:
  report       Path to an existing full .ncu-rep file.
  kernel_name  Function-form kernel name shown by NCU.
  invocation   Matching kernel invocation. Defaults to 6 because
               benchmark/profile_entry.py performs 5 warmup launches.

Environment:
  NCU_BIN      Optional path to the ncu executable.

Notes:
  The report must have been collected with --set full. Metrics not present in
  the report are printed as N/A instead of being treated as zero.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

REPORT=$1
KERNEL=$2
INVOCATION=${3:-6}

if [[ ! -f "$REPORT" ]]; then
  echo "error: report not found: $REPORT" >&2
  exit 1
fi

if [[ ! "$INVOCATION" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: invocation must be a positive integer: $INVOCATION" >&2
  exit 2
fi

find_ncu() {
  if [[ -n ${NCU_BIN:-} ]]; then
    printf '%s\n' "$NCU_BIN"
    return
  fi

  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return
  fi

  local candidate
  for candidate in \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-12.6/bin/ncu \
    /usr/local/cuda-12.4/bin/ncu \
    /usr/local/cuda-12.1/bin/ncu; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

if ! NCU=$(find_ncu); then
  echo "error: ncu was not found; add it to PATH or set NCU_BIN" >&2
  exit 1
fi

if [[ ! -x "$NCU" ]]; then
  echo "error: ncu is not executable: $NCU" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

KERNEL_ID="::${KERNEL}:${INVOCATION}"

# Use one regex so unavailable architecture-specific metrics are omitted by NCU
# instead of making the whole import fail. The parser reports omitted values as
# N/A, which is different from a measured value of zero.
METRICS='regex:^(smsp__sass_inst_executed_op_shared_(ld|st)\.sum|l1tex__data_pipe_lsu_wavefronts_mem_shared_op_(ld|st)\.sum|l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_(ld|st)\.sum|l1tex__t_(requests|sectors)_pipe_lsu_mem_(global|local)_op_(ld|st)\.sum|lts__t_(bytes|sectors)\.sum|lts__t_sector_hit_rate\.pct|dram__bytes_(read|write)\.sum|(sm|smsp)__pipe_fma_cycles_active\.avg\.pct_of_peak_sustained_active|sm__pipe_tensor.*\.avg\.pct_of_peak_sustained_active|smsp__inst_executed_pipe_tensor\.sum|smsp__sass_thread_inst_executed_op_hmma_pred_on\.sum|smsp__sass_thread_inst_executed_op_(mufu|fadd|fmul|ffma|branch)_pred_on\.sum|smsp__sass_average_branch_targets_threads_uniform\.pct|sm__maximum_warps_per_active_cycle_pct|sm__blocks_active\.avg\.per_cycle_active|launch__occupancy_limit_blocks|launch__waves_per_multiprocessor|smsp__warps_eligible\.avg\.per_cycle_active|smsp__(average_warps_issue_stalled_(barrier|long_scoreboard|short_scoreboard|mio_throttle|lg_throttle)_per_issue_active\.ratio|warp_issue_stalled_(barrier|long_scoreboard|short_scoreboard|mio_throttle|lg_throttle)_per_warp_active\.pct)|smsp__sass_inst_executed_op_local_(ld|st)\.sum)$'

if ! "$NCU" \
  --import "$REPORT" \
  --page raw \
  --kernel-name-base function \
  --kernel-id "$KERNEL_ID" \
  --metrics "$METRICS" \
  >"$TMP_DIR/raw.txt"; then
  echo "error: failed to import supplemental metrics from $REPORT" >&2
  exit 1
fi

awk '
  function save(metric, key) {
    value[key] = $NF
    unit[key] = $2
    numeric[key] = $NF
    gsub(/,/, "", numeric[key])
    numeric[key] += 0
  }

  function shown(key, fallback_unit) {
    if (!(key in value)) {
      return "N/A"
    }

    current_unit = unit[key]
    if (current_unit == "" || current_unit == value[key]) {
      current_unit = fallback_unit
    }

    return current_unit == "" ? value[key] : value[key] " " current_unit
  }

  function as_bytes(key, current_unit) {
    if (!(key in numeric)) {
      return -1
    }

    current_unit = unit[key]
    if (current_unit == "Kbyte") return numeric[key] * 1000
    if (current_unit == "Mbyte") return numeric[key] * 1000000
    if (current_unit == "Gbyte") return numeric[key] * 1000000000
    if (current_unit == "Tbyte") return numeric[key] * 1000000000000
    return numeric[key]
  }

  function human_bytes(bytes) {
    if (bytes < 0) return "N/A"
    if (bytes >= 1000000000) return sprintf("%.2f Gbyte", bytes / 1000000000)
    if (bytes >= 1000000) return sprintf("%.2f Mbyte", bytes / 1000000)
    if (bytes >= 1000) return sprintf("%.2f Kbyte", bytes / 1000)
    return sprintf("%.0f byte", bytes)
  }

  function pair(load_key, store_key, fallback_unit) {
    return "load=" shown(load_key, fallback_unit) "; store=" shown(store_key, fallback_unit)
  }

  $1 == "smsp__sass_inst_executed_op_shared_ld.sum" {
    save($1, "shared_ld_inst")
  }
  $1 == "smsp__sass_inst_executed_op_shared_st.sum" {
    save($1, "shared_st_inst")
  }
  $1 == "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum" {
    save($1, "shared_ld_wavefront")
  }
  $1 == "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum" {
    save($1, "shared_st_wavefront")
  }
  $1 == "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum" {
    save($1, "shared_ld_conflict")
  }
  $1 == "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum" {
    save($1, "shared_st_conflict")
  }
  $1 == "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" {
    save($1, "global_ld_request")
  }
  $1 == "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum" {
    save($1, "global_st_request")
  }
  $1 == "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" {
    save($1, "global_ld_sector")
  }
  $1 == "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum" {
    save($1, "global_st_sector")
  }
  $1 == "lts__t_bytes.sum" {
    save($1, "l2_bytes")
  }
  $1 == "lts__t_sectors.sum" {
    save($1, "l2_sectors")
  }
  $1 == "lts__t_sector_hit_rate.pct" {
    save($1, "l2_hit_rate")
  }
  $1 == "dram__bytes_read.sum" {
    save($1, "dram_read_bytes")
  }
  $1 == "dram__bytes_write.sum" {
    save($1, "dram_write_bytes")
  }
  $1 == "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" ||
  $1 == "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" {
    save($1, "fma_util")
  }
  $1 ~ /^sm__pipe_tensor.*\.avg\.pct_of_peak_sustained_active$/ {
    save($1, "tensor_util")
  }
  $1 == "smsp__inst_executed_pipe_tensor.sum" {
    save($1, "tensor_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum" {
    save($1, "hmma_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_mufu_pred_on.sum" {
    save($1, "mufu_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum" {
    save($1, "fadd_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum" {
    save($1, "fmul_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum" {
    save($1, "ffma_inst")
  }
  $1 == "smsp__sass_thread_inst_executed_op_branch_pred_on.sum" {
    save($1, "branch_inst")
  }
  $1 == "smsp__sass_average_branch_targets_threads_uniform.pct" {
    save($1, "branch_uniform")
  }
  $1 == "sm__maximum_warps_per_active_cycle_pct" {
    save($1, "theoretical_occupancy")
  }
  $1 == "launch__waves_per_multiprocessor" {
    save($1, "waves_per_sm")
  }
  $1 == "sm__blocks_active.avg.per_cycle_active" {
    save($1, "active_blocks_per_sm")
  }
  $1 == "launch__occupancy_limit_blocks" {
    save($1, "occupancy_limit_blocks")
  }
  $1 == "smsp__warps_eligible.avg.per_cycle_active" {
    save($1, "eligible_warps")
  }
  $1 ~ /^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$/ {
    stall = $1
    sub(/^smsp__average_warps_issue_stalled_/, "", stall)
    sub(/_per_issue_active\.ratio$/, "", stall)
    save($1, stall "_cycles")
  }
  $1 ~ /^smsp__warp_issue_stalled_.*_per_warp_active\.pct$/ {
    stall = $1
    sub(/^smsp__warp_issue_stalled_/, "", stall)
    sub(/_per_warp_active\.pct$/, "", stall)
    save($1, stall "_pct")
  }
  $1 == "smsp__sass_inst_executed_op_local_ld.sum" {
    save($1, "local_ld_inst")
  }
  $1 == "smsp__sass_inst_executed_op_local_st.sum" {
    save($1, "local_st_inst")
  }
  $1 == "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum" {
    save($1, "local_ld_request")
  }
  $1 == "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum" {
    save($1, "local_st_request")
  }

  END {
    if ("l2_bytes" in value) {
      l2_text = shown("l2_bytes", "byte")
    } else if ("l2_sectors" in numeric) {
      l2_text = human_bytes(numeric["l2_sectors"] * 32)
    } else {
      l2_text = "N/A"
    }

    if ("dram_read_bytes" in value && "dram_write_bytes" in value) {
      dram_total = as_bytes("dram_read_bytes") + as_bytes("dram_write_bytes")
      dram_text = "L2 total=" l2_text \
        "; DRAM read=" shown("dram_read_bytes", "byte") \
        "; write=" shown("dram_write_bytes", "byte") \
        "; total=" human_bytes(dram_total)
    } else {
      dram_text = "L2 total=" l2_text \
        "; DRAM read=" shown("dram_read_bytes", "byte") \
        "; write=" shown("dram_write_bytes", "byte") \
        "; total=N/A"
    }

    local_text = "instructions: " pair("local_ld_inst", "local_st_inst", "inst") \
      "; requests: " pair("local_ld_request", "local_st_request", "request")

    printf "| %-34s | %-96s |\n", "Supplemental Metric", "Value"
    printf "| %-34s | %-96s |\n", "----------------------------------", "------------------------------------------------------------------------------------------------"
    printf "| %-34s | %-96s |\n", "Shared Load Instructions", shown("shared_ld_inst", "inst")
    printf "| %-34s | %-96s |\n", "Shared Store Instructions", shown("shared_st_inst", "inst")
    printf "| %-34s | %-96s |\n", "Shared Load Wavefronts", shown("shared_ld_wavefront", "wavefront")
    printf "| %-34s | %-96s |\n", "Shared Store Wavefronts", shown("shared_st_wavefront", "wavefront")
    printf "| %-34s | %-96s |\n", "Shared Bank Conflicts", pair("shared_ld_conflict", "shared_st_conflict", "conflict")
    printf "| %-34s | %-96s |\n", "Global Requests", pair("global_ld_request", "global_st_request", "request")
    printf "| %-34s | %-96s |\n", "Global Sectors", pair("global_ld_sector", "global_st_sector", "sector")
    printf "| %-34s | %-96s |\n", "L2 / DRAM Bytes", dram_text
    printf "| %-34s | %-96s |\n", "L2 Hit Rate", shown("l2_hit_rate", "%")
    printf "| %-34s | %-96s |\n", "FMA Pipe Utilization", shown("fma_util", "%")
    printf "| %-34s | %-96s |\n", "Tensor Pipe Utilization", shown("tensor_util", "%")
    printf "| %-34s | %-96s |\n", "Tensor / HMMA Instructions", "tensor=" shown("tensor_inst", "inst") "; HMMA=" shown("hmma_inst", "inst")
    printf "| %-34s | %-96s |\n", "MUFU / SFU Instructions", shown("mufu_inst", "inst")
    printf "| %-34s | %-96s |\n", "FP Arithmetic Instructions", "FADD=" shown("fadd_inst", "inst") "; FMUL=" shown("fmul_inst", "inst") "; FFMA=" shown("ffma_inst", "inst")
    printf "| %-34s | %-96s |\n", "Branch Instructions", shown("branch_inst", "inst")
    printf "| %-34s | %-96s |\n", "Uniform Branch Targets", shown("branch_uniform", "%")
    printf "| %-34s | %-96s |\n", "Theoretical Occupancy", shown("theoretical_occupancy", "%")
    printf "| %-34s | %-96s |\n", "Active Blocks / SM", shown("active_blocks_per_sm", "block/SM")
    printf "| %-34s | %-96s |\n", "Occupancy Block Limit", shown("occupancy_limit_blocks", "block")
    printf "| %-34s | %-96s |\n", "Waves Per SM", shown("waves_per_sm", "wave/SM")
    printf "| %-34s | %-96s |\n", "Eligible Warps / Scheduler", shown("eligible_warps", "warp/cycle")
    printf "| %-34s | %-96s |\n", "Barrier Stall", "cycles=" shown("barrier_cycles", "cycle") "; percent=" shown("barrier_pct", "%")
    printf "| %-34s | %-96s |\n", "Long Scoreboard Stall", "cycles=" shown("long_scoreboard_cycles", "cycle") "; percent=" shown("long_scoreboard_pct", "%")
    printf "| %-34s | %-96s |\n", "Short Scoreboard Stall", "cycles=" shown("short_scoreboard_cycles", "cycle") "; percent=" shown("short_scoreboard_pct", "%")
    printf "| %-34s | %-96s |\n", "MIO Throttle Stall", "cycles=" shown("mio_throttle_cycles", "cycle") "; percent=" shown("mio_throttle_pct", "%")
    printf "| %-34s | %-96s |\n", "LG Throttle Stall", "cycles=" shown("lg_throttle_cycles", "cycle") "; percent=" shown("lg_throttle_pct", "%")
    printf "| %-34s | %-96s |\n", "Local Load/Store", local_text
  }
' "$TMP_DIR/raw.txt" >"$TMP_DIR/values.txt"

printf 'Report: %s\n' "$REPORT"
printf 'Kernel: %s (invocation %s)\n\n' "$KERNEL" "$INVOCATION"
cat "$TMP_DIR/values.txt"

if grep -q 'N/A' "$TMP_DIR/values.txt"; then
  cat >&2 <<'EOF'

warning: N/A means the metric was not found in this report; it does not mean 0.
         Confirm that the report was collected with --set full and that the
         kernel name and invocation select the intended launch.
EOF
fi
