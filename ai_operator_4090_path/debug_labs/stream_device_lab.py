#!/usr/bin/env python3
"""Reproducible Stream/Event/Device experiments for aiop4090."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import aiop4090 as ops


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def delayed_fill(tensor: torch.Tensor, value: float, sleep_cycles: int) -> None:
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError("this PyTorch build does not expose torch.cuda._sleep")
    torch.cuda._sleep(sleep_cycles)
    tensor.fill_(value)


def run_dependency(use_event: bool, sleep_cycles: int) -> dict[str, Any]:
    size = 64
    a = torch.zeros(size, size, device="cuda")
    b = torch.eye(size, device="cuda")
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    ready = torch.cuda.Event()

    with torch.cuda.stream(producer):
        delayed_fill(a, 3.0, sleep_cycles)
        ready.record()

    with torch.cuda.stream(consumer):
        if use_event:
            consumer.wait_event(ready)
        out = ops.gemm_naive(a, b)

    consumer.synchronize()
    producer.synchronize()
    expected = torch.full_like(out, 3.0)
    max_error = (out - expected).abs().max().item()
    correct = bool(torch.equal(out, expected))
    return {
        "event_dependency": use_event,
        "correct": correct,
        "max_abs_error": max_error,
        "producer_stream": int(producer.cuda_stream),
        "consumer_stream": int(consumer.cuda_stream),
    }


def current_stream(sleep_cycles: int) -> dict[str, Any]:
    result = run_dependency(use_event=True, sleep_cycles=sleep_cycles)
    if not result["correct"]:
        raise RuntimeError("operator did not honor the consumer current stream after wait_event")
    result["status"] = "PASS"
    result["conclusion"] = "CUDAGuard/current-stream path honors the explicit Event dependency"
    return result


def missing_event(sleep_cycles: int) -> dict[str, Any]:
    result = run_dependency(use_event=False, sleep_cycles=sleep_cycles)
    result["status"] = "EXPECTED_RACE_REPRODUCED" if not result["correct"] else "INCONCLUSIVE"
    result["conclusion"] = (
        "consumer observed stale data because no Event dependency existed"
        if not result["correct"] else
        "scheduling hid the race; increase --sleep-cycles and rerun"
    )
    return result


def fixed_event(sleep_cycles: int) -> dict[str, Any]:
    result = run_dependency(use_event=True, sleep_cycles=sleep_cycles)
    if not result["correct"]:
        raise RuntimeError("wait_event did not repair the producer/consumer dependency")
    result["status"] = "PASS"
    result["conclusion"] = "Event makes the cross-stream producer/consumer dependency explicit"
    return result


def device_guard() -> dict[str, Any]:
    count = torch.cuda.device_count()
    if count < 2:
        return {"status": "N/A", "reason": "only one GPU is visible"}
    original = torch.cuda.current_device()
    target = 1 if original == 0 else 0
    with torch.cuda.device(original):
        a = torch.randn(17, 19, device=f"cuda:{target}")
        b = torch.randn(19, 13, device=f"cuda:{target}")
        out = ops.gemm_naive(a, b)
        torch.cuda.synchronize(target)
    if out.device.index != target or not torch.allclose(out, a @ b, atol=2e-4, rtol=2e-4):
        raise RuntimeError("CUDAGuard did not keep allocation and launch on the input device")
    return {
        "status": "PASS",
        "current_device_before_call": original,
        "input_device": target,
        "output_device": out.device.index,
    }


def wrong_device() -> dict[str, Any]:
    if torch.cuda.device_count() < 2:
        return {"status": "N/A", "reason": "only one GPU is visible"}
    a = torch.randn(8, 8, device="cuda:0")
    b = torch.randn(8, 8, device="cuda:1")
    try:
        ops.gemm_naive(a, b)
    except RuntimeError as error:
        if "same CUDA device" not in str(error):
            raise
        return {"status": "PASS", "error": str(error), "layer": "API contract before launch"}
    raise RuntimeError("mixed-device inputs were not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["current-stream", "missing-event", "fixed-event",
                                           "device-guard", "wrong-device"], required=True)
    parser.add_argument("--sleep-cycles", type=int, default=200_000_000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    require_cuda()
    runners = {
        "current-stream": lambda: current_stream(args.sleep_cycles),
        "missing-event": lambda: missing_event(args.sleep_cycles),
        "fixed-event": lambda: fixed_event(args.sleep_cycles),
        "device-guard": device_guard,
        "wrong-device": wrong_device,
    }
    result = {"case": args.case, **runners[args.case]()}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.case == "missing-event" and result["status"] == "INCONCLUSIVE":
        print(
            "missing-event race was not reproduced; increase --sleep-cycles and rerun",
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
