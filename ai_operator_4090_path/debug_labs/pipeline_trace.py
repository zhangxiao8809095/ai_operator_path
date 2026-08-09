#!/usr/bin/env python3
"""NVTX-instrumented end-to-end paths for Python-to-kernel attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import torch

import aiop4090 as ops


def nvtx(name: str):
    return torch.cuda.nvtx.range(name)


def make_case(scenario: str, size: int) -> tuple[Callable[[], torch.Tensor], torch.Tensor]:
    torch.manual_seed(2026)
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    reference = a @ b

    if scenario == "baseline":
        def run():
            with nvtx("operator_call/gemm_tiled"):
                return ops.gemm_tiled(a, b)
    elif scenario == "hidden-copy":
        a_view = a.t()
        b_view = b.t()
        reference = a_view @ b_view

        def run():
            with nvtx("hidden_overhead/contiguous"):
                a_ready = a_view.contiguous()
                b_ready = b_view.contiguous()
            with nvtx("operator_call/gemm_tiled"):
                return ops.gemm_tiled(a_ready, b_ready)
    elif scenario == "hidden-sync":
        def run():
            with nvtx("operator_call/gemm_tiled"):
                out = ops.gemm_tiled(a, b)
            with nvtx("hidden_overhead/synchronize"):
                torch.cuda.synchronize()
            return out
    elif scenario == "wmma-fp16-input":
        a_half = a.half()
        b_half = b.half()
        reference = (a_half @ b_half).float()

        def run():
            with nvtx("operator_call/wmma_fp16_input"):
                return ops.gemm_wmma_fp16(a_half, b_half)
    elif scenario == "wmma-fp32-input":
        def run():
            with nvtx("operator_call/wmma_fp32_internal_conversion"):
                return ops.gemm_wmma_fp16(a, b)
    else:
        raise ValueError(scenario)
    return run, reference


def execute(scenario: str, size: int, iters: int) -> dict[str, object]:
    run, reference = make_case(scenario, size)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    samples = []
    out = None
    for index in range(iters):
        start = time.perf_counter()
        with nvtx(f"iteration/{scenario}/{index}"):
            out = run()
        samples.append((time.perf_counter() - start) * 1e3)
    torch.cuda.synchronize()
    assert out is not None
    tolerance = 1.0 if scenario.startswith("wmma") else 3e-3
    max_error = (out - reference).abs().max().item()
    if not torch.allclose(out, reference, atol=tolerance, rtol=2e-2):
        raise RuntimeError(f"correctness failed before profiling: max_abs_error={max_error}")
    return {
        "scenario": scenario,
        "size": size,
        "iters": iters,
        "cpu_submission_median_ms": statistics.median(samples),
        "cpu_submission_min_ms": min(samples),
        "cpu_submission_max_ms": max(samples),
        "max_abs_error": max_error,
        "note": "Use NSYS NVTX ranges for layer attribution; CPU submission is not kernel Duration.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["baseline", "hidden-copy", "hidden-sync",
                                               "wmma-fp16-input", "wmma-fp32-input"], required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = execute(args.scenario, args.size, args.iters)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
