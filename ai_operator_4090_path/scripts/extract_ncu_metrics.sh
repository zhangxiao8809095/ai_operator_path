#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  extract_ncu_metrics.sh <report.ncu-rep> <kernel_name> [invocation]

Examples:
  extract_ncu_metrics.sh \
    reports/ncu/gemm_naive_full.ncu-rep \
    gemm_naive_kernel \
    6

  extract_ncu_metrics.sh \
    reports/ncu/gemm_tiled_full.ncu-rep \
    gemm_tiled_kernel \
    6

Arguments:
  report       Path to an existing .ncu-rep file.
  kernel_name  Function-form kernel name shown by NCU.
  invocation   Matching kernel invocation to read. Defaults to 6 because
               benchmark/profile_entry.py performs 5 warmup launches.

Environment:
  NCU_BIN      Optional path to the ncu executable.
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

BASIC_METRICS='gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,regex:^(dram__throughput|gpu__dram_throughput)\.avg\.pct_of_peak_sustained_elapsed$,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread'

if ! "$NCU" \
  --import "$REPORT" \
  --page raw \
  --kernel-name-base function \
  --kernel-id "$KERNEL_ID" \
  --metrics "$BASIC_METRICS" \
  >"$TMP_DIR/basic.txt"; then
  echo "error: failed to import basic metrics from $REPORT" >&2
  exit 1
fi

awk '
  $1 == "gpu__time_duration.sum" {
    value["Duration"] = $NF
    unit["Duration"] = $2
  }
  $1 == "sm__throughput.avg.pct_of_peak_sustained_elapsed" {
    value["Compute Throughput"] = $NF
    unit["Compute Throughput"] = $2
  }
  $1 == "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" {
    value["Memory Throughput"] = $NF
    unit["Memory Throughput"] = $2
  }
  $1 ~ /^(dram__throughput|gpu__dram_throughput)\.avg\.pct_of_peak_sustained_elapsed$/ {
    value["DRAM Throughput"] = $NF
    unit["DRAM Throughput"] = $2
  }
  $1 == "lts__throughput.avg.pct_of_peak_sustained_elapsed" {
    value["L2 Throughput"] = $NF
    unit["L2 Throughput"] = $2
  }
  $1 == "sm__warps_active.avg.pct_of_peak_sustained_active" {
    value["Achieved Occupancy"] = $NF
    unit["Achieved Occupancy"] = $2
  }
  $1 == "launch__registers_per_thread" {
    value["Registers / Thread"] = $NF
    unit["Registers / Thread"] = $2
  }
  END {
    names[1] = "Duration"
    names[2] = "Compute Throughput"
    names[3] = "Memory Throughput"
    names[4] = "DRAM Throughput"
    names[5] = "L2 Throughput"
    names[6] = "Achieved Occupancy"
    names[7] = "Registers / Thread"

    found = 0
    for (i = 1; i <= 7; ++i) {
      name = names[i]
      if (name in value) {
        printf "%s\t%s %s\n", name, value[name], unit[name]
        ++found
      } else {
        printf "%s\tN/A\n", name
      }
    }

    if (found == 0) {
      exit 3
    }
  }
' "$TMP_DIR/basic.txt" >"$TMP_DIR/basic_values.txt" || {
  status=$?
  if (( status == 3 )); then
    echo "error: no matching metrics were found" >&2
    echo "hint: check the kernel name and invocation number" >&2
    echo "hint: inspect available launches with:" >&2
    echo "  $NCU --import '$REPORT' --page raw --csv" >&2
  else
    echo "error: failed to parse basic NCU metrics" >&2
  fi
  exit "$status"
}

STALL_METRICS='regex:^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$,regex:^smsp__warp_issue_stalled_.*_per_warp_active\.pct$'

if ! "$NCU" \
  --import "$REPORT" \
  --page raw \
  --kernel-name-base function \
  --kernel-id "$KERNEL_ID" \
  --metrics "$STALL_METRICS" \
  >"$TMP_DIR/stalls.txt"; then
  echo "error: failed to import warp stall metrics from $REPORT" >&2
  exit 1
fi

awk '
  /^smsp__average_warps_issue_stalled_/ {
    name = $1
    sub(/^smsp__average_warps_issue_stalled_/, "", name)
    sub(/_per_issue_active\.ratio$/, "", name)
    cycles[name] = $NF + 0
  }
  /^smsp__warp_issue_stalled_/ && $1 ~ /\.pct$/ {
    name = $1
    sub(/^smsp__warp_issue_stalled_/, "", name)
    sub(/_per_warp_active\.pct$/, "", name)
    percent[name] = $NF + 0
  }
  END {
    top = ""
    max_cycles = -1

    for (name in cycles) {
      if (name != "selected" && cycles[name] > max_cycles) {
        top = name
        max_cycles = cycles[name]
      }
    }

    if (top == "") {
      max_percent = -1
      for (name in percent) {
        if (name != "selected" && percent[name] > max_percent) {
          top = name
          max_percent = percent[name]
        }
      }
    }

    display["lg_throttle"] = "LG Throttle"
    display["mio_throttle"] = "MIO Throttle"
    display["long_scoreboard"] = "Long Scoreboard"
    display["short_scoreboard"] = "Short Scoreboard"
    display["math_pipe_throttle"] = "Math Pipe Throttle"
    display["barrier"] = "Barrier"
    display["membar"] = "Membar"
    display["not_selected"] = "Not Selected"
    display["wait"] = "Wait"

    if (top == "") {
      print "N/A"
      exit
    }

    label = (top in display) ? display[top] : top
    if ((top in cycles) && (top in percent)) {
      printf "%s (%.2f cycles, %.2f%%)\n", label, cycles[top], percent[top]
    } else if (top in cycles) {
      printf "%s (%.2f cycles)\n", label, cycles[top]
    } else {
      printf "%s (%.2f%%)\n", label, percent[top]
    }
  }
' "$TMP_DIR/stalls.txt" >"$TMP_DIR/top_stall.txt"

printf 'Report: %s\n' "$REPORT"
printf 'Kernel: %s (invocation %s)\n\n' "$KERNEL" "$INVOCATION"
printf '| %-28s | %-52s |\n' 'Metric' 'Value'
printf '| %-28s | %-52s |\n' '----------------------------' '----------------------------------------------------'

while IFS=$'\t' read -r name value; do
  printf '| %-28s | %-52s |\n' "$name" "$value"
done <"$TMP_DIR/basic_values.txt"

TOP_STALL=$(<"$TMP_DIR/top_stall.txt")
printf '| %-28s | %-52s |\n' 'Top Stall Reason' "$TOP_STALL"
