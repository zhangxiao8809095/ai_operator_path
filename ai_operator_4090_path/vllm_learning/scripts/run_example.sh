#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

lesson="${1:-}"
shift || true

case "${lesson}" in
  basic)
    exec python examples/01_basic_inference.py "$@"
    ;;
  batch)
    exec python examples/02_offline_batch.py "$@"
    ;;
  sampling)
    exec python examples/03_sampling_params.py "$@"
    ;;
  kv-cache)
    exec python examples/04_kv_cache_observe.py "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_example.sh {basic|batch|sampling|kv-cache} [args...]"
    exit 2
    ;;
esac
