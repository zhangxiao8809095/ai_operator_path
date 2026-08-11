#!/usr/bin/env python3
"""Host/static and RTX 4090 preflight checks for engineering/debug experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENTS = [
    "ENG-C01", "ENG-C02", "ENG-C03", "ENG-D01", "ENG-P01",
    "DBG-E01", "DBG-E02", "DBG-E03", "DBG-E04",
    "DBG-S01", "DBG-S02", "DBG-S03", "DBG-S04",
    "DBG-L01", "DBG-L02", "DBG-L03", "DBG-L04",
    "DBG-T01", "DBG-T02", "DBG-T03",
    "DBG-U01", "DBG-U02", "DBG-U03", "DBG-U04",
]
EXPECTED_EXPORTS = [
    "gemm_naive", "gemm_tiled", "gemm_tiled_padding", "gemm_regtile2x2",
    "gemm_regtile4x4", "gemm_vectorized_float4", "gemm_wmma_fp16",
    "softmax_row", "softmax_block_reduce", "softmax_warp_reduce", "softmax_online",
    "layernorm_row", "layernorm_block_reduce", "layernorm_warp_reduce", "layernorm_vectorized",
    "rmsnorm_row", "rmsnorm_block_reduce", "rmsnorm_warp_reduce", "rmsnorm_vectorized",
    "rmsnorm_vectorized_float4", "attention_naive", "attention_causal_naive",
    "attention_kv_cache_decode", "attention_tiled_online_softmax",
]


def find_tool(name: str, cuda_candidates: bool = False, override_var: str | None = None) -> str | None:
    if override_var and os.environ.get(override_var):
        configured = Path(os.environ[override_var]).expanduser()
        if configured.is_file():
            return str(configured)
    discovered = shutil.which(name)
    if discovered:
        return discovered
    if cuda_candidates:
        if os.environ.get("CUDA_HOME"):
            candidate = Path(os.environ["CUDA_HOME"]) / "bin" / name
            if candidate.is_file():
                return str(candidate)
        for cuda_root in ("cuda", "cuda-12.6", "cuda-12.4", "cuda-12.1"):
            candidate = Path("/usr/local") / cuda_root / "bin" / name
            if candidate.is_file():
                return str(candidate)
    return None


def host_checks() -> tuple[dict[str, Any], list[str]]:
    required_files = [
        "scripts/00_check_env.sh",
        "scripts/10_build.sh",
        "scripts/15_verify_4090.sh",
        "scripts/40_sanitize.sh",
        "scripts/50_debug_labs.sh",
        "scripts/run_gemm_experiment.sh",
        "scripts/run_operator_experiment.sh",
        "scripts/run_debug_experiment.sh",
        "scripts/verify_workspace.py",
        "debug_labs/build_fault_lab.py",
        "debug_labs/diagnose_extension.py",
        "debug_labs/pipeline_trace.py",
        "debug_labs/run_fault_lab.py",
        "debug_labs/stream_device_lab.py",
        "debug_labs/unknown_fault_lab.py",
        "debug_labs/fault_extension/bindings.cpp",
        "debug_labs/fault_extension/faults.cu",
        "debug_labs/fault_extension/setup.py",
        "docs/operator_validation_experiments_4090.md",
        "tests/test_operator_validation.py",
        "tests/test_sanitizer_smoke.py",
    ]
    missing_files = [relative for relative in required_files if not (ROOT / relative).is_file()]
    executable_files = [
        "scripts/00_check_env.sh",
        "scripts/10_build.sh",
        "scripts/15_verify_4090.sh",
        "scripts/40_sanitize.sh",
        "scripts/50_debug_labs.sh",
        "scripts/run_gemm_experiment.sh",
        "scripts/run_operator_experiment.sh",
        "scripts/run_debug_experiment.sh",
        "scripts/verify_workspace.py",
        "debug_labs/preflight.py",
    ]
    non_executable_files = [
        relative for relative in executable_files if not os.access(ROOT / relative, os.X_OK)
    ]
    runner_path = ROOT / "scripts/run_debug_experiment.sh"
    runner_text = runner_path.read_text(encoding="utf-8") if runner_path.is_file() else ""
    missing_experiments = [name for name in EXPECTED_EXPERIMENTS if name not in runner_text]
    handbook_path = ROOT / "docs/operator_validation_experiments_4090.md"
    handbook_text = handbook_path.read_text(encoding="utf-8") if handbook_path.is_file() else ""
    documented_experiments = re.findall(r"^(?:###|####) ((?:ENG|DBG)-[A-Z]\d{2})：", handbook_text, re.MULTILINE)
    missing_documented = sorted(set(EXPECTED_EXPERIMENTS) - set(documented_experiments))
    unexpected_documented = sorted(set(documented_experiments) - set(EXPECTED_EXPERIMENTS))
    wrong_command_entries = [
        name
        for name in EXPECTED_EXPERIMENTS
        if f"`bash scripts/run_debug_experiment.sh {name}`" not in handbook_text
    ]

    fault_source = (ROOT / "debug_labs/fault_extension/faults.cu")
    fault_text = fault_source.read_text(encoding="utf-8") if fault_source.is_file() else ""
    expected_faults = (
        "out_of_bounds_kernel",
        "shared_race_kernel",
        "uninitialized_read_kernel",
        "illegal_address_kernel",
    )
    missing_faults = [name for name in expected_faults if name not in fault_text]

    result = {
        "project_root": str(ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "expected_experiment_count": len(EXPECTED_EXPERIMENTS),
        "missing_files": missing_files,
        "non_executable_files": non_executable_files,
        "missing_experiment_entries": missing_experiments,
        "documented_experiment_count": len(documented_experiments),
        "missing_documented_experiments": missing_documented,
        "unexpected_documented_experiments": unexpected_documented,
        "wrong_document_commands": wrong_command_entries,
        "missing_fault_kernels": missing_faults,
    }
    failures = []
    if missing_files:
        failures.append(f"required files missing: {missing_files}")
    if non_executable_files:
        failures.append(f"required entrypoints are not executable: {non_executable_files}")
    if missing_experiments:
        failures.append(f"runner entries missing: {missing_experiments}")
    if missing_documented or unexpected_documented:
        failures.append(
            f"handbook experiment IDs differ: missing={missing_documented}, "
            f"unexpected={unexpected_documented}"
        )
    if wrong_command_entries:
        failures.append(f"handbook unified commands missing: {wrong_command_entries}")
    if missing_faults:
        failures.append(f"fault kernels missing: {missing_faults}")
    return result, failures


def gpu_checks() -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {}
    failures: list[str] = []
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on the GPU host
        return result, [f"PyTorch import failed: {error}"]

    result.update({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_sleep_available": hasattr(torch.cuda, "_sleep"),
    })
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        result.update({
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "capability": list(capability),
        })
        if capability != (8, 9):
            failures.append(
                f"expected RTX 4090 compute capability 8.9, got {capability[0]}.{capability[1]}"
            )
        if not hasattr(torch.cuda, "_sleep"):
            failures.append("torch.cuda._sleep is unavailable; deterministic stream-race lab cannot run")

    tools = {
        "nvcc": find_tool("nvcc", cuda_candidates=True),
        "ncu": find_tool("ncu", cuda_candidates=True, override_var="NCU_BIN"),
        "nsys": find_tool("nsys", cuda_candidates=True, override_var="NSYS_BIN"),
        "compute-sanitizer": find_tool(
            "compute-sanitizer", cuda_candidates=True, override_var="COMPUTE_SANITIZER_BIN"
        ),
        "cuobjdump": find_tool("cuobjdump", cuda_candidates=True, override_var="CUOBJDUMP_BIN"),
        "ldd": find_tool("ldd"),
        "cxx": find_tool("c++") or find_tool("g++"),
    }
    result["tools"] = tools
    for name, path in tools.items():
        if path is None:
            failures.append(f"required tool not found: {name}")

    try:
        import aiop4090
        import aiop4090._C as extension
    except Exception as error:  # pragma: no cover - exercised on the GPU host
        failures.append(f"aiop4090 extension import failed: {error}")
    else:
        missing_python = [name for name in EXPECTED_EXPORTS if not callable(getattr(aiop4090, name, None))]
        missing_binding = [name for name in EXPECTED_EXPORTS if not callable(getattr(extension, name, None))]
        result.update({
            "extension_path": str(Path(extension.__file__).resolve()),
            "python_export_count": len(EXPECTED_EXPORTS) - len(missing_python),
            "binding_export_count": len(EXPECTED_EXPORTS) - len(missing_binding),
            "missing_python_exports": missing_python,
            "missing_binding_exports": missing_binding,
        })
        if missing_python or missing_binding:
            failures.append(
                f"formal exports incomplete: python={missing_python}, binding={missing_binding}"
            )
        cuobjdump = tools.get("cuobjdump")
        if cuobjdump:
            completed = subprocess.run(
                [cuobjdump, "--list-elf", str(Path(extension.__file__).resolve())],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            result["cuobjdump_status"] = completed.returncode
            result["sm_89_cubin"] = "sm_89" in completed.stdout
            if completed.returncode != 0 or "sm_89" not in completed.stdout:
                failures.append("formal extension does not expose an sm_89 cubin")
    return result, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-only", action="store_true", help="skip CUDA, tool and extension checks")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    host_result, failures = host_checks()
    result: dict[str, Any] = {"host": host_result}
    if not args.host_only:
        gpu_result, gpu_failures = gpu_checks()
        result["gpu"] = gpu_result
        failures.extend(gpu_failures)
    result["status"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
