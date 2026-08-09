#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

EXPERIMENT=${1:-help}
REPORT_DIR="$PROJECT_DIR/reports/debug_labs"
mkdir -p "$REPORT_DIR"

find_python_bin() {
  if [[ -n ${PYTHON_BIN:-} ]]; then
    local configured
    configured=$(command -v "$PYTHON_BIN" 2>/dev/null || true)
    if [[ -z "$configured" && -x "$PYTHON_BIN" ]]; then
      configured=$PYTHON_BIN
    fi
    [[ -n "$configured" ]] || return 1
    printf '%s\n' "$configured"
    return
  fi
  local name candidate
  for name in python python3; do
    candidate=$(command -v "$name" 2>/dev/null || true)
    if [[ -x "$candidate" ]] && "$candidate" -c \
      'import sys; raise SystemExit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
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
export PYTHON_BIN

usage() {
  cat <<'EOF'
Usage: bash scripts/run_debug_experiment.sh <experiment-or-group>

Experiments:
  ENG-C01 ENG-C02 ENG-C03 ENG-D01 ENG-P01
  DBG-E01 DBG-E02 DBG-E03 DBG-E04
  DBG-S01 DBG-S02 DBG-S03 DBG-S04
  DBG-L01 DBG-L02 DBG-L03 DBG-L04
  DBG-T01 DBG-T02 DBG-T03
  DBG-U01 DBG-U02 DBG-U03 DBG-U04

Groups:
  preflight engineering extension stream pipeline sanitizer unknown all

Notes:
  - Run preflight first on the RTX 4090 server.
  - Intentional CUDA faults always run in isolated Python processes.
  - DBG-S04 records N/A and returns success when only one GPU is visible.
EOF
}

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

run_helper() {
  bash scripts/50_debug_labs.sh "$@"
}

run_nsys_pair() {
  local log_file=$1
  local first=$2
  local second=$3
  run_log "$log_file" run_helper nsys "$first"
  append_log "$log_file" run_helper nsys "$second"
}

run_named() {
  local id=$1
  local log="$REPORT_DIR/${id}.log"
  case "$id" in
    ENG-C01)
      run_log "$log" "$PYTHON_BIN" -m pytest -q -s \
        tests/test_operator_validation.py -k all_exports_use_current_stream
      ;;
    ENG-C02)
      run_log "$log" "$PYTHON_BIN" -m pytest -q -s \
        tests/test_operator_validation.py \
        -k 'contract_rejects_cpu_wrong_dtype_and_noncontiguous or attention_rejects_invalid_contracts'
      ;;
    ENG-C03)
      run_log "$log" run_helper stream device-guard
      append_log "$log" run_helper stream wrong-device
      ;;
    ENG-D01)
      run_log "$log" run_helper launch
      append_log "$log" run_helper async-error
      append_log "$log" run_helper memcheck
      ;;
    ENG-P01)
      run_log "$log" run_helper nsys baseline
      for scenario in hidden-copy hidden-sync wmma-fp16-input wmma-fp32-input; do
        append_log "$log" run_helper nsys "$scenario"
      done
      ;;
    DBG-E01)
      run_log "$log" run_helper diagnose
      ;;
    DBG-E02)
      run_log "$log" run_helper build-integration
      ;;
    DBG-E03)
      run_log "$log" run_helper build-faults
      ;;
    DBG-E04)
      run_log "$log" "$PYTHON_BIN" debug_labs/build_fault_lab.py --case missing-export
      append_log "$log" "$PYTHON_BIN" debug_labs/build_fault_lab.py --case undefined-symbol
      ;;
    DBG-S01)
      run_log "$log" run_helper stream current-stream
      ;;
    DBG-S02)
      run_log "$log" run_helper stream missing-event
      ;;
    DBG-S03)
      run_log "$log" run_helper stream fixed-event
      ;;
    DBG-S04)
      run_log "$log" run_helper stream device-guard
      append_log "$log" run_helper stream wrong-device
      ;;
    DBG-L01)
      run_nsys_pair "$log" baseline hidden-copy
      ;;
    DBG-L02)
      run_nsys_pair "$log" baseline hidden-sync
      ;;
    DBG-L03)
      run_nsys_pair "$log" wmma-fp16-input wmma-fp32-input
      ;;
    DBG-L04)
      run_log "$log" run_helper launch
      append_log "$log" run_helper memcheck
      ;;
    DBG-T01)
      run_log "$log" run_helper memcheck
      ;;
    DBG-T02)
      run_log "$log" run_helper racecheck
      ;;
    DBG-T03)
      run_log "$log" run_helper initcheck
      ;;
    DBG-U01|DBG-U02|DBG-U03|DBG-U04)
      local case_id=${id#DBG-}
      run_log "$log" run_helper unknown "$case_id"
      ;;
    *)
      return 2
      ;;
  esac
}

run_group() {
  local id
  for id in "$@"; do
    echo "===== $id ====="
    "$SCRIPT_DIR/run_debug_experiment.sh" "$id"
  done
}

case "$EXPERIMENT" in
  help|-h|--help)
    usage
    ;;
  preflight)
    run_log "$REPORT_DIR/preflight.log" run_helper preflight
    ;;
  engineering)
    run_group ENG-C01 ENG-C02 ENG-C03 ENG-D01 ENG-P01
    ;;
  extension)
    run_group DBG-E01 DBG-E02 DBG-E03 DBG-E04
    ;;
  stream)
    run_group DBG-S01 DBG-S02 DBG-S03 DBG-S04
    ;;
  pipeline)
    run_group DBG-L01 DBG-L02 DBG-L03 DBG-L04
    ;;
  sanitizer)
    run_group DBG-T01 DBG-T02 DBG-T03
    ;;
  unknown)
    run_group DBG-U01 DBG-U02 DBG-U03 DBG-U04
    ;;
  all)
    "$SCRIPT_DIR/run_debug_experiment.sh" preflight
    run_group \
      ENG-C01 ENG-C02 ENG-C03 ENG-D01 ENG-P01 \
      DBG-E01 DBG-E02 DBG-E03 DBG-E04 \
      DBG-S01 DBG-S02 DBG-S03 DBG-S04 \
      DBG-L01 DBG-L02 DBG-L03 DBG-L04 \
      DBG-T01 DBG-T02 DBG-T03 \
      DBG-U01 DBG-U02 DBG-U03 DBG-U04
    ;;
  ENG-C01|ENG-C02|ENG-C03|ENG-D01|ENG-P01|\
  DBG-E01|DBG-E02|DBG-E03|DBG-E04|\
  DBG-S01|DBG-S02|DBG-S03|DBG-S04|\
  DBG-L01|DBG-L02|DBG-L03|DBG-L04|\
  DBG-T01|DBG-T02|DBG-T03|\
  DBG-U01|DBG-U02|DBG-U03|DBG-U04)
    run_named "$EXPERIMENT"
    ;;
  *)
    echo "unknown debug experiment or group: $EXPERIMENT" >&2
    usage >&2
    exit 2
    ;;
esac
