#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${PYTHON:-}" ]]; then
  python_command="${PYTHON}"
elif [[ -x .venv/bin/python ]]; then
  python_command=".venv/bin/python"
else
  python_command="python3"
fi

experiment="${1:-}"
if [[ -z "${experiment}" ]]; then
  echo "usage: bash scripts/run_experiment.sh <name> [arguments...]" >&2
  echo "names: preflight, engine-lifecycle, prefill-decode, offline-batching," >&2
  echo "       sampling, kv-pressure, service-smoke, continuous-batching," >&2
  echo "       chunked-prefill, prefix-caching, render-results" >&2
  exit 2
fi
shift

case "${experiment}" in
  preflight)             program="experiments/00_preflight.py" ;;
  engine-lifecycle)      program="experiments/01_engine_lifecycle.py" ;;
  prefill-decode)        program="experiments/02_prefill_decode_sweep.py" ;;
  offline-batching)      program="experiments/03_offline_batching.py" ;;
  sampling)              program="experiments/04_sampling_diagnostics.py" ;;
  kv-pressure)           program="experiments/05_kv_pressure.py" ;;
  service-smoke)         program="experiments/06_service_smoke.py" ;;
  continuous-batching)   program="experiments/07_continuous_batching.py" ;;
  chunked-prefill)       program="experiments/08_chunked_prefill.py" ;;
  prefix-caching)        program="experiments/09_prefix_caching.py" ;;
  render-results)        program="experiments/render_results.py" ;;
  *)
    echo "unknown experiment: ${experiment}" >&2
    exit 2
    ;;
esac

exec "${python_command}" "${program}" "$@"
