#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

model="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_key="${VLLM_API_KEY:-local-token}"
served_model="${VLLM_SERVED_MODEL_NAME:-vllm-lab}"
dtype="${VLLM_DTYPE:-auto}"
tp_size="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
max_model_len="${VLLM_MAX_MODEL_LEN:-4096}"
trust_remote_code="${VLLM_TRUST_REMOTE_CODE:-false}"

extra_args=()
case "${trust_remote_code}" in
  1|true|TRUE|True|yes|YES|Yes|on|ON|On)
    extra_args+=(--trust-remote-code)
    ;;
  0|false|FALSE|False|no|NO|No|off|OFF|Off)
    ;;
  *)
    echo "VLLM_TRUST_REMOTE_CODE must be true or false, got: ${trust_remote_code}" >&2
    exit 2
    ;;
esac

exec vllm serve "${model}" \
  --host "${host}" \
  --port "${port}" \
  --api-key "${api_key}" \
  --served-model-name "${served_model}" \
  --dtype "${dtype}" \
  --tensor-parallel-size "${tp_size}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --generation-config vllm \
  "${extra_args[@]}"
