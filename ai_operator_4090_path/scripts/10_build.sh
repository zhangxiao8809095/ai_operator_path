#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip setuptools wheel ninja pytest
# 开发阶段建议不使用 build isolation，减少 PyTorch/CUDA extension 构建问题。
python -m pip install -e . --no-build-isolation
