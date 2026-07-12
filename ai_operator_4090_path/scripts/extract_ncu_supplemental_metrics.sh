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
    /usr/local/cuda-12.6/bin/ncu; do
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
METRICS='regex:^(smsp__sass_inst_executed_op_shared_(ld|st)\.sum|l1tex__data_pipe_lsu_wavefronts_mem_shared_op_(ld|st)\.sum|l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_(ld|st)\.sum|l1tex__t_requests_pipe_lsu_mem_(global|local)_op_(ld|st)\.sum|lts__t_bytes\.sum|dram__bytes_(read|write)\.sum|smsp__pipe_fma_cycles_active\.avg\.pct_of_peak_sustained_active|sm__maximum_warps_per_active_cycle_pct|launch__waves_per_multiprocessor|smsp__warps_eligible\.avg\.per_cycle_active|smsp__sass_inst_executed_op_local_(ld|st)\.sum)$'

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

  function shown_as(key, output_unit) {
    if (!(key in value)) {
      return "N/A"
    }
    return value[key] " " output_unit
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
  $1 == "lts__t_bytes.sum" {
    save($1, "l2_bytes")
  }
  $1 == "dram__bytes_read.sum" {
    save($1, "dram_read_bytes")
  }
  $1 == "dram__bytes_write.sum" {
    save($1, "dram_write_bytes")
  }
  $1 == "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" {
    save($1, "fma_util")
  }
  $1 == "sm__maximum_warps_per_active_cycle_pct" {
    save($1, "theoretical_occupancy")
  }
  $1 == "launch__waves_per_multiprocessor" {
    save($1, "waves_per_sm")
  }
  $1 == "smsp__warps_eligible.avg.per_cycle_active" {
    save($1, "eligible_warps")
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
    if ("dram_read_bytes" in numeric && "dram_write_bytes" in numeric) {
      dram_total = numeric["dram_read_bytes"] + numeric["dram_write_bytes"]
      dram_text = "L2 total=" shown("l2_bytes", "byte") \
        "; DRAM read=" shown("dram_read_bytes", "byte") \
        "; write=" shown("dram_write_bytes", "byte") \
        "; total=" sprintf("%.0f byte", dram_total)
    } else {
      dram_text = "L2 total=" shown("l2_bytes", "byte") \
        "; DRAM read=" shown("dram_read_bytes", "byte") \
        "; write=" shown("dram_write_bytes", "byte") \
        "; total=N/A"
    }

    local_text = "instructions: " pair("local_ld_inst", "local_st_inst", "inst") \
      "; requests: " pair("local_ld_request", "local_st_request", "request")

    printf "| %-34s | %-96s |\n", "Supplemental Metric", "Value"
    printf "| %-34s | %-96s |\n", "----------------------------------", "------------------------------------------------------------------------------------------------"
    printf "| %-34s | %-96s |\n", "Shared Load Instructions", shown("shared_ld_inst", "inst")
    printf "| %-34s | %-96s |\n", "Shared Load Requests", shown_as("shared_ld_inst", "request")
    printf "| %-34s | %-96s |\n", "Shared Load Wavefronts", shown("shared_ld_wavefront", "wavefront")
    printf "| %-34s | %-96s |\n", "Shared Bank Conflicts", pair("shared_ld_conflict", "shared_st_conflict", "conflict")
    printf "| %-34s | %-96s |\n", "Global Load Requests", shown("global_ld_request", "request")
    printf "| %-34s | %-96s |\n", "L2 / DRAM Bytes", dram_text
    printf "| %-34s | %-96s |\n", "FMA Pipe Utilization", shown("fma_util", "%")
    printf "| %-34s | %-96s |\n", "Theoretical Occupancy", shown("theoretical_occupancy", "%")
    printf "| %-34s | %-96s |\n", "Waves Per SM", shown("waves_per_sm", "wave/SM")
    printf "| %-34s | %-96s |\n", "Eligible Warps / Scheduler", shown("eligible_warps", "warp/cycle")
    printf "| %-34s | %-96s |\n", "Local Load/Store", local_text
  }
' "$TMP_DIR/raw.txt" >"$TMP_DIR/values.txt"

printf 'Report: %s\n' "$REPORT"
printf 'Kernel: %s (invocation %s)\n\n' "$KERNEL" "$INVOCATION"
cat "$TMP_DIR/values.txt"

printf '\nShared Load Requests equals Shared Load Instructions on RTX 4090 (SM 8.9):\n'
printf 'each regular shared-memory instruction generates one request.\n'

if grep -q 'N/A' "$TMP_DIR/values.txt"; then
  cat >&2 <<'EOF'

warning: N/A means the metric was not found in this report; it does not mean 0.
         Confirm that the report was collected with --set full and that the
         kernel name and invocation select the intended launch.
EOF
fi
