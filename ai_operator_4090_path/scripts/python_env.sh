#!/usr/bin/env bash

# Shared Python resolver for the RTX 4090 scripts.  This file is intended to be
# sourced after PROJECT_DIR has been set by the caller.

python_env_resolve_executable() {
  local candidate=${1:-}
  local resolved=""
  [[ -n "$candidate" ]] || return 1
  resolved=$(command -v "$candidate" 2>/dev/null || true)
  if [[ -z "$resolved" && -x "$candidate" ]]; then
    resolved=$candidate
  fi
  [[ -n "$resolved" && -x "$resolved" ]] || return 1
  printf '%s\n' "$resolved"
}

python_env_is_python3() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info[0] != 3)' \
    >/dev/null 2>&1
}

python_env_has_cuda_torch() {
  "$1" -c \
    'import torch; raise SystemExit(torch.version.cuda is None)' \
    >/dev/null 2>&1
}

python_env_candidates() {
  local candidate resolved
  local candidates=()

  if [[ -n ${VIRTUAL_ENV:-} ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi
  if [[ -n ${PROJECT_DIR:-} ]]; then
    candidates+=("$PROJECT_DIR/.venv/bin/python")
  fi
  if [[ -n ${CONDA_PREFIX:-} ]]; then
    candidates+=("$CONDA_PREFIX/bin/python")
  fi

  candidates+=(python python3)

  # Common locations used by GPU rental images. They are considered only when
  # the files exist and never override an explicitly configured PYTHON_BIN.
  candidates+=(
    /opt/conda/bin/python
    /root/miniconda3/bin/python
    /root/anaconda3/bin/python
  )

  for candidate in "${candidates[@]}"; do
    if resolved=$(python_env_resolve_executable "$candidate"); then
      printf '%s\n' "$resolved"
    fi
  done
}

find_python_bin() {
  local requirement=${1:-python3}
  local candidate fallback="" seen=""

  # An explicit override is authoritative. Do not silently switch to another
  # environment when the caller intentionally selected one.
  if [[ -n ${PYTHON_BIN:-} ]]; then
    candidate=$(python_env_resolve_executable "$PYTHON_BIN") || return 1
    python_env_is_python3 "$candidate" || return 1
    if [[ "$requirement" == "cuda-torch" ]]; then
      python_env_has_cuda_torch "$candidate" || return 1
    fi
    printf '%s\n' "$candidate"
    return
  fi

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    case $seen in
      *"|$candidate|"*) continue ;;
    esac
    seen+="|$candidate|"
    python_env_is_python3 "$candidate" || continue
    [[ -n "$fallback" ]] || fallback=$candidate
    if [[ "$requirement" != "cuda-torch" ]] || \
       python_env_has_cuda_torch "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(python_env_candidates)

  if [[ "$requirement" != "cuda-torch" && -n "$fallback" ]]; then
    printf '%s\n' "$fallback"
    return
  fi
  return 1
}

show_python_candidates() {
  local candidate seen=""
  echo "Python candidates checked:"
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    case $seen in
      *"|$candidate|"*) continue ;;
    esac
    seen+="|$candidate|"
    if ! python_env_is_python3 "$candidate"; then
      echo "  $candidate: not Python 3"
    elif python_env_has_cuda_torch "$candidate"; then
      "$candidate" -c \
        'import torch; print("  " + __import__("sys").executable + ": torch " + torch.__version__ + ", CUDA " + str(torch.version.cuda))'
    elif "$candidate" -c 'import torch' >/dev/null 2>&1; then
      "$candidate" -c \
        'import torch; print("  " + __import__("sys").executable + ": torch " + torch.__version__ + ", CPU-only")'
    else
      echo "  $candidate: torch not installed"
    fi
  done < <(python_env_candidates)
}

show_cuda_torch_remediation() {
  local selected=${1:-none}
  cat >&2 <<EOF
error: no selected Python environment contains CUDA-enabled PyTorch.
selected Python: $selected

If the rental image already contains PyTorch, activate that environment or run:
  PYTHON_BIN=/absolute/path/to/that/python bash scripts/15_verify_4090.sh smoke

If no Python environment contains PyTorch, create an isolated environment and
install the official CUDA wheel matching the server. For this project's CUDA
12.6 path, the usual commands are:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Then verify before rebuilding:
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
  PYTHON_BIN="\$(pwd)/.venv/bin/python" bash scripts/15_verify_4090.sh smoke

Do not install torch into /usr/bin/python3 with sudo pip. Keep this CUDA
extension environment separate from vllm_learning.
EOF
}
