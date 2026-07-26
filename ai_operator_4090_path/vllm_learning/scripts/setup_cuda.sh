#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.12 --seed --managed-python
fi

source .venv/bin/activate

torch_backend="${VLLM_TORCH_BACKEND:-cu126}"
if [[ -n "${VLLM_SPEC:-}" ]]; then
  uv pip install --torch-backend="${torch_backend}" "${VLLM_SPEC}"
else
  uv pip install \
    --torch-backend=cu126 \
    --extra-index-url https://wheels.vllm.ai/0.10.0/cu126 \
    -r requirements/vllm.txt
fi
uv pip install -e ".[client,dev]"

python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
PY
