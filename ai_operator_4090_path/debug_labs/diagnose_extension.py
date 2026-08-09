#!/usr/bin/env python3
"""Diagnose aiop4090 build/import integration without changing the installation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPORTS = [
    "gemm_naive", "gemm_tiled", "gemm_tiled_padding", "gemm_regtile2x2",
    "gemm_regtile4x4", "gemm_vectorized_float4", "gemm_wmma_fp16",
    "softmax_row", "softmax_block_reduce", "softmax_warp_reduce", "softmax_online",
    "layernorm_row", "layernorm_block_reduce", "layernorm_warp_reduce", "layernorm_vectorized",
    "rmsnorm_row", "rmsnorm_block_reduce", "rmsnorm_warp_reduce", "rmsnorm_vectorized",
    "rmsnorm_vectorized_float4", "attention_naive", "attention_causal_naive",
    "attention_kv_cache_decode", "attention_tiled_online_softmax",
]


def newest_source() -> tuple[Path, float]:
    candidates = [ROOT / "setup.py", ROOT / "pyproject.toml", ROOT / "src/aiop4090/__init__.py"]
    candidates.extend((ROOT / "src/aiop4090/csrc").glob("*"))
    files = [path for path in candidates if path.is_file()]
    newest = max(files, key=lambda path: path.stat().st_mtime)
    return newest, newest.stat().st_mtime


def command_output(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    return completed.returncode, completed.stdout.strip()


def diagnose(include_loader: bool) -> tuple[dict[str, Any], list[str]]:
    package = importlib.import_module("aiop4090")
    extension = importlib.import_module("aiop4090._C")
    extension_path = Path(extension.__file__).resolve()
    source_path, source_mtime = newest_source()
    extension_mtime = extension_path.stat().st_mtime
    missing_python = [name for name in EXPECTED_EXPORTS if not callable(getattr(package, name, None))]
    missing_binding = [name for name in EXPECTED_EXPORTS if not callable(getattr(extension, name, None))]

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "torch_build_config": torch.__config__.show(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>"),
        "package_path": str(Path(package.__file__).resolve()),
        "extension_path": str(extension_path),
        "extension_mtime": extension_mtime,
        "newest_source": str(source_path),
        "newest_source_mtime": source_mtime,
        "source_newer_than_extension": source_mtime > extension_mtime + 1.0,
        "python_exports": len(EXPECTED_EXPORTS) - len(missing_python),
        "binding_exports": len(EXPECTED_EXPORTS) - len(missing_binding),
        "missing_python_exports": missing_python,
        "missing_binding_exports": missing_binding,
    }

    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        result.update({
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(capability),
        })
        if capability != (8, 9):
            failures.append(f"expected RTX 4090 capability 8.9, got {capability[0]}.{capability[1]}")

    if result["source_newer_than_extension"]:
        failures.append("a source/build file is newer than the loaded extension; clean rebuild required")
    if missing_python:
        failures.append(f"Python wrappers missing: {missing_python}")
    if missing_binding:
        failures.append(f"binding exports missing: {missing_binding}")

    if include_loader and shutil.which("ldd"):
        status, output = command_output(["ldd", str(extension_path)])
        result["ldd_status"] = status
        result["ldd_output"] = output
        if status != 0 or "not found" in output:
            failures.append("dynamic loader dependencies are unresolved; inspect ldd_output")

    if include_loader and shutil.which("cuobjdump"):
        status, output = command_output(["cuobjdump", "--list-elf", str(extension_path)])
        result["cuobjdump_status"] = status
        result["cuobjdump_output"] = output
        if status == 0 and torch.cuda.is_available() and "sm_89" not in output:
            failures.append("loaded extension does not expose an sm_89 cubin in cuobjdump output")

    return result, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the diagnostic record to this path")
    parser.add_argument("--strict", action="store_true", help="return non-zero when a risk is detected")
    parser.add_argument("--loader", action="store_true", help="also inspect ldd and cuobjdump when available")
    args = parser.parse_args()

    result, failures = diagnose(args.loader)
    result["failures"] = failures
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
