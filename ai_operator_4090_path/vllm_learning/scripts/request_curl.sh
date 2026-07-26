#!/usr/bin/env bash
set -euo pipefail

host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_key="${VLLM_API_KEY:-local-token}"
served_model="${VLLM_SERVED_MODEL_NAME:-vllm-lab}"

curl --fail-with-body --silent --show-error \
  "http://${host}:${port}/v1/chat/completions" \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${served_model}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"你是一位简洁的 vLLM 教学助手。\"},
      {\"role\": \"user\", \"content\": \"一句话说明 continuous batching 的价值。\"}
    ],
    \"temperature\": 0.2,
    \"max_tokens\": 128
  }"
echo
