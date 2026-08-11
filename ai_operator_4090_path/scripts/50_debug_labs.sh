#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPORT_DIR="$ROOT_DIR/reports/debug_labs"
FAULT_DIR="$ROOT_DIR/debug_labs/fault_extension"
mkdir -p "$REPORT_DIR"
cd "$ROOT_DIR"

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

find_cuda_tool() {
  local name=$1
  local override=${2:-}
  if [[ -n "$override" && -x "$override" ]]; then
    printf '%s\n' "$override"
    return
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return
  fi
  local candidate
  if [[ -n ${CUDA_HOME:-} && -x "$CUDA_HOME/bin/$name" ]]; then
    printf '%s\n' "$CUDA_HOME/bin/$name"
    return
  fi
  for candidate in \
    "/usr/local/cuda/bin/$name" \
    "/usr/local/cuda-12.6/bin/$name" \
    "/usr/local/cuda-12.4/bin/$name" \
    "/usr/local/cuda-12.1/bin/$name"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/50_debug_labs.sh <command> [argument]

Commands:
  preflight                Verify files, RTX 4090, tools and all 24 formal exports.
  diagnose                 Inspect loaded extension, symbols, loader and sm_89 code.
  build-integration        Run safe stale/missing-symbol/source/arch diagnostic cases.
  stream <case>            current-stream | missing-event | fixed-event | device-guard | wrong-device
  pipeline <scenario>      baseline | hidden-copy | hidden-sync | wmma-fp16-input | wmma-fp32-input
  nsys <scenario>          Profile one pipeline scenario with CUDA and NVTX.
  build-faults             Build the isolated intentional-fault extension.
  launch                   Verify an invalid launch is reported at the launcher.
  async-error              Compare an execution fault with CUDA_LAUNCH_BLOCKING=0/1.
  memcheck                 Detect the intentional out-of-bounds write.
  racecheck                Detect the intentional shared-memory race.
  initcheck                Detect the intentional uninitialized read.
  unknown [U01|U02|U03|U04] Run a blind root-cause case; default is deterministic random.
EOF
}

build_faults() {
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
  export MAX_JOBS=${MAX_JOBS:-2}
  (
    cd "$FAULT_DIR"
    "$PYTHON_BIN" setup.py build_ext --inplace
  )
  local module_path cuobjdump_output
  module_path=$("$PYTHON_BIN" -c \
    'import pathlib, sys; sys.path.insert(0, "debug_labs/fault_extension"); import aiop4090_faults; print(pathlib.Path(aiop4090_faults.__file__).resolve())')
  if ! CUOBJDUMP=$(find_cuda_tool cuobjdump "${CUOBJDUMP_BIN:-}"); then
    echo "cuobjdump was not found; add it to PATH or set CUOBJDUMP_BIN" >&2
    exit 1
  fi
  cuobjdump_output=$("$CUOBJDUMP" --list-elf "$module_path")
  if ! grep -q 'sm_89' <<<"$cuobjdump_output"; then
    echo "fault extension does not contain an sm_89 cubin: $module_path" >&2
    exit 1
  fi
  echo "Fault extension: $module_path"
  echo "Verified fault-extension cubin: sm_89"
}

run_sanitizer_fault() {
  local tool=$1
  local fault_case=$2
  local log="$REPORT_DIR/${tool}_${fault_case}.log"
  local kernel_pattern evidence_pattern
  if ! COMPUTE_SANITIZER=$(find_cuda_tool compute-sanitizer "${COMPUTE_SANITIZER_BIN:-}"); then
    echo "compute-sanitizer was not found; add it to PATH or set COMPUTE_SANITIZER_BIN" >&2
    exit 1
  fi
  build_faults
  set +e
  "$COMPUTE_SANITIZER" --tool "$tool" --error-exitcode 86 \
    "$PYTHON_BIN" debug_labs/run_fault_lab.py --case "$fault_case" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -eq 0 ]]; then
    echo "Expected $tool to detect the intentional fault, but it returned success." >&2
    exit 1
  fi
  if [[ $status -ne 86 ]]; then
    echo "$tool failed unexpectedly with status=$status; inspect $log" >&2
    exit "$status"
  fi
  case "$tool:$fault_case" in
    memcheck:oob)
      kernel_pattern='out_of_bounds_kernel'
      evidence_pattern='invalid.*(global|__global__).*write|out of bounds'
      ;;
    racecheck:race)
      kernel_pattern='shared_race_kernel'
      evidence_pattern='race reported|hazard|WAW|RAW'
      ;;
    initcheck:init)
      kernel_pattern='uninitialized_read_kernel'
      evidence_pattern='uninitialized.*(read|memory)'
      ;;
    *)
      echo "error: no sanitizer evidence rule for $tool:$fault_case" >&2
      exit 2
      ;;
  esac
  if ! grep -q "$kernel_pattern" "$log" || ! grep -Eqi "$evidence_pattern" "$log"; then
    echo "expected $tool evidence for $kernel_pattern was not found in $log" >&2
    exit 1
  fi
  echo "PASS: $tool detected the intentional fault. Report: $log"
}

run_async_error_comparison() {
  local mode status log
  local summary="$REPORT_DIR/async_error_summary.log"
  build_faults
  : >"$summary"
  for mode in 0 1; do
    log="$REPORT_DIR/async_error_blocking_${mode}.log"
    set +e
    CUDA_LAUNCH_BLOCKING="$mode" \
      "$PYTHON_BIN" debug_labs/run_fault_lab.py --case illegal-address \
      >"$log" 2>&1
    status=$?
    set -e
    cat "$log"
    if [[ $status -eq 0 ]]; then
      echo "error: illegal-address unexpectedly succeeded with CUDA_LAUNCH_BLOCKING=$mode" >&2
      exit 1
    fi
    if ! grep -Eqi 'illegal memory access|illegal address|CUDA error' "$log"; then
      echo "error: expected CUDA illegal-address evidence was not found in $log" >&2
      exit 1
    fi
    printf 'CUDA_LAUNCH_BLOCKING=%s status=%s log=%s\n' "$mode" "$status" "$log" \
      | tee -a "$summary"
  done
  echo "PASS: both isolated processes exposed the intentional execution error."
  echo "Compare the first aiop4090/fault-extension frame in the two logs: $summary"
}

COMMAND=${1:-help}
ARGUMENT=${2:-}
case "$COMMAND" in
  help|-h|--help)
    usage
    ;;
  preflight)
    "$PYTHON_BIN" debug_labs/preflight.py \
      --json "$REPORT_DIR/preflight.json"
    ;;
  diagnose)
    "$PYTHON_BIN" debug_labs/diagnose_extension.py --strict --loader \
      --json "$REPORT_DIR/extension_diagnostic.json"
    ;;
  build-integration)
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
    "$PYTHON_BIN" debug_labs/build_fault_lab.py --case all | tee "$REPORT_DIR/build_integration.log"
    ;;
  stream)
    [[ -n "$ARGUMENT" ]] || { echo "stream requires a case" >&2; exit 2; }
    "$PYTHON_BIN" debug_labs/stream_device_lab.py --case "$ARGUMENT" \
      --sleep-cycles "${STREAM_SLEEP_CYCLES:-200000000}" \
      --json "$REPORT_DIR/stream_${ARGUMENT}.json"
    ;;
  pipeline)
    [[ -n "$ARGUMENT" ]] || { echo "pipeline requires a scenario" >&2; exit 2; }
    "$PYTHON_BIN" debug_labs/pipeline_trace.py --scenario "$ARGUMENT" \
      --json "$REPORT_DIR/pipeline_${ARGUMENT}.json"
    ;;
  nsys)
    [[ -n "$ARGUMENT" ]] || { echo "nsys requires a scenario" >&2; exit 2; }
    if ! NSYS=$(find_cuda_tool nsys "${NSYS_BIN:-}"); then
      echo "nsys was not found; add it to PATH or set NSYS_BIN" >&2
      exit 1
    fi
    "$NSYS" profile --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true \
      -o "$REPORT_DIR/pipeline_${ARGUMENT}" \
      "$PYTHON_BIN" debug_labs/pipeline_trace.py --scenario "$ARGUMENT" \
      --json "$REPORT_DIR/pipeline_${ARGUMENT}.json"
    ;;
  build-faults)
    build_faults
    "$PYTHON_BIN" debug_labs/run_fault_lab.py --case identity
    ;;
  launch)
    build_faults
    "$PYTHON_BIN" debug_labs/run_fault_lab.py --case invalid-launch | tee "$REPORT_DIR/invalid_launch.log"
    ;;
  async-error)
    run_async_error_comparison
    ;;
  memcheck)
    run_sanitizer_fault memcheck oob
    ;;
  racecheck)
    run_sanitizer_fault racecheck race
    ;;
  initcheck)
    run_sanitizer_fault initcheck init
    ;;
  unknown)
    case_id=${ARGUMENT:-random}
    "$PYTHON_BIN" debug_labs/unknown_fault_lab.py --case "$case_id" \
      --json "$REPORT_DIR/unknown_${case_id}.json"
    ;;
  *)
    echo "unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
