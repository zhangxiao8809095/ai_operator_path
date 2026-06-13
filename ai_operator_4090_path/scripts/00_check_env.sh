#!/usr/bin/env bash
set -euo pipefail

echo "[1] GPU"
nvidia-smi || true

echo "\n[2] CUDA compiler"
nvcc --version || true

echo "\n[3] Nsight"
which ncu && ncu --version || true
which nsys && nsys --version || true

echo "\n[4] Python / PyTorch"
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
