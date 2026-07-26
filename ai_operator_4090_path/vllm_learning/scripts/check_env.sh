#!/usr/bin/env bash
set -euo pipefail

echo "[OS]"
if [[ -r /etc/os-release ]]; then
  sed -n '1,6p' /etc/os-release
else
  uname -a
fi

echo
echo "[Python 3.12]"
if command -v python3.12 >/dev/null 2>&1; then
  python3.12 --version
else
  echo "NOT FOUND: install Python 3.12 or let uv download a managed Python."
fi

echo
echo "[NVIDIA driver and GPU]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
    --format=csv,noheader
else
  echo "NOT FOUND: nvidia-smi (this machine cannot run the CUDA lessons)."
fi

echo
echo "[Local CUDA toolkit]"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found. Prebuilt wheels can still run; nvcc is needed for source builds."
fi

echo
echo "[Installed Python packages]"
python_command=""
if [[ -x .venv/bin/python ]]; then
  python_command=".venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  python_command="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  python_command="python3"
fi

if [[ -n "${python_command}" ]]; then
  "${python_command}" - <<'PY'
import importlib.util
import platform

print("python:", platform.python_version())
for package in ("torch", "vllm", "openai", "pytest", "ruff"):
    print(f"{package}:", "installed" if importlib.util.find_spec(package) else "missing")
PY
fi
