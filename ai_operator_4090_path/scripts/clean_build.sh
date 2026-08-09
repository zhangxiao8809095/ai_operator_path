#!/usr/bin/env bash
set -euo pipefail
rm -rf build dist *.egg-info src/*.egg-info src/aiop4090.egg-info
find . -name "*.so" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
