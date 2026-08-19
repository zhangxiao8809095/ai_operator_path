#!/usr/bin/env python3
"""Experiment 00: verify the exact Python, vLLM, CUDA and GPU runtime."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-non-4090", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/00_preflight.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig.from_env()

    import torch
    import vllm

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    checks = [
        {
            "name": "python_3_12",
            "passed": sys.version_info[:2] == (3, 12),
            "actual": platform.python_version(),
        },
        {
            "name": "vllm_0_10_0",
            "passed": vllm.__version__.startswith("0.10.0"),
            "actual": vllm.__version__,
        },
        {
            "name": "pytorch_cuda_available",
            "passed": cuda_available,
            "actual": cuda_available,
        },
        {
            "name": "pytorch_cuda_12_6",
            "passed": bool(torch.version.cuda and torch.version.cuda.startswith("12.6")),
            "actual": torch.version.cuda,
        },
        {
            "name": "enough_visible_gpus_for_tp",
            "passed": device_count >= config.tensor_parallel_size,
            "actual": {"device_count": device_count, "tp": config.tensor_parallel_size},
        },
        {
            "name": "rtx_4090_visible",
            "passed": args.allow_non_4090 or any("4090" in name for name in gpu_names),
            "actual": gpu_names,
        },
    ]

    devices = []
    if cuda_available:
        for index in range(device_count):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "free_gib": free_bytes / 1024**3,
                    "total_gib": total_bytes / 1024**3,
                }
            )

    passed = all(item["passed"] for item in checks)
    report = {
        "experiment_id": "00_preflight",
        "title": "环境与版本分层检查",
        "status": "passed" if passed else "failed",
        "config": config.llm_kwargs(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "devices": devices,
        },
        "checks": checks,
    }
    output = write_json(args.output, report)
    for item in checks:
        marker = "PASS" if item["passed"] else "FAIL"
        print(f"[{marker}] {item['name']}: {item['actual']}")
    print(f"Report: {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
