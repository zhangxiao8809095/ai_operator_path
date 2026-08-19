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

echo "[1/5] Compile Python sources"
"${python_command}" -m compileall -q src examples experiments tests

echo "[2/5] Import GPU-independent package"
"${python_command}" -c "from vllm_lab import LabConfig; print(LabConfig.from_env())"

echo "[3/5] Run unit tests"
"${python_command}" -m unittest discover -s tests -v

echo "[4/5] Check shell syntax"
for script in scripts/*.sh; do
  bash -n "${script}"
done

echo "[5/5] Run Ruff when available"
if "${python_command}" -m ruff --version >/dev/null 2>&1; then
  "${python_command}" -m ruff check src examples experiments tests
else
  echo "SKIP: Ruff is not installed in this environment."
fi

"${python_command}" - <<'PY'
import importlib.util

if importlib.util.find_spec("vllm"):
    import vllm

    print("vLLM import OK:", vllm.__version__)
else:
    print("vLLM import SKIPPED: install GPU dependencies on the CUDA server.")
PY
