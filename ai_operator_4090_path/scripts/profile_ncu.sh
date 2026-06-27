#!/usr/bin/env bash
set -e

OP=${1:-gemm_naive}
ITERS=${ITERS:-20}
OUT_DIR=${OUT_DIR:-reports/ncu}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [[ ! -x "$CUDA_HOME/bin/ncu" && -d /usr/local/cuda-12.6 ]]; then
  CUDA_HOME=/usr/local/cuda-12.6
fi

mkdir -p "$OUT_DIR"

TORCH_LIB=$($PYTHON_BIN -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

sudo env \
  PATH="$CUDA_HOME/bin:$PATH" \
  LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
  TORCH_CUDA_ARCH_LIST="8.9" \
  ncu --target-processes all \
      --set speed-of-light \
      --force-overwrite \
      -o "$OUT_DIR/${OP}_sol" \
      "$PYTHON_BIN" benchmark/profile_entry.py \
      --op "$OP" \
      --iters "$ITERS"
