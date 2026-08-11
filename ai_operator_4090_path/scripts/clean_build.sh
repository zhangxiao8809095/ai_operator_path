#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

rm -rf build dist *.egg-info src/*.egg-info src/aiop4090.egg-info
rm -rf debug_labs/fault_extension/build debug_labs/fault_extension/*.egg-info
find src/aiop4090 -maxdepth 1 -type f -name "_C*.so" -delete
find debug_labs/fault_extension -maxdepth 1 -type f -name "aiop4090_faults*.so" -delete
find benchmark debug_labs src tests -name "__pycache__" -type d -prune -exec rm -rf {} +
