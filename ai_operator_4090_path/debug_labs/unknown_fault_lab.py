#!/usr/bin/env python3
"""Small blind debugging cases. Use --reveal only after writing a root-cause report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable

import torch

import aiop4090 as ops
from stream_device_lab import missing_event


ANSWERS = {
    "U01": "numerical: unstable softmax computes exp(x) without subtracting the row maximum",
    "U02": "async/stream: producer and consumer streams have no Event dependency",
    "U03": "API/binding: float16 violates the current softmax float32 contract",
    "U04": "performance/integration: a synchronize call turns asynchronous submission into per-call blocking",
}


def u01() -> dict[str, Any]:
    x = torch.tensor([[10_000.0, 9_999.0, -10_000.0]], device="cuda")
    bad = torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)
    repeated = torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)
    ref = torch.softmax(x, dim=-1)
    return {
        "symptom": "finite input produced non-finite output",
        "actual": bad.detach().cpu().tolist(),
        "reference": ref.detach().cpu().tolist(),
        "repeat_pattern_stable": bool(torch.allclose(bad, repeated, equal_nan=True)),
    }


def u02() -> dict[str, Any]:
    base_sleep_cycles = int(os.environ.get("STREAM_SLEEP_CYCLES", "200000000"))
    attempts = []
    result = None
    for multiplier in (1, 2, 4):
        sleep_cycles = base_sleep_cycles * multiplier
        result = missing_event(sleep_cycles)
        attempts.append({"sleep_cycles": sleep_cycles, "status": result["status"]})
        if result["status"] == "EXPECTED_RACE_REPRODUCED":
            break
    assert result is not None
    if result["status"] != "EXPECTED_RACE_REPRODUCED":
        raise RuntimeError(
            "U02 stream race remained inconclusive after three increasing delays; "
            "set STREAM_SLEEP_CYCLES to a larger value and rerun"
        )
    return {
        "symptom": "same tensors produce stale output under two non-default streams",
        "sleep_cycles": sleep_cycles,
        "attempts": attempts,
        "max_abs_error": result["max_abs_error"],
        "status": result["status"],
    }


def u03() -> dict[str, Any]:
    x = torch.randn(4, 33, device="cuda", dtype=torch.float16)
    try:
        ops.softmax_row(x)
    except RuntimeError as error:
        return {"symptom": "call failed before a useful output was produced", "error": str(error)}
    raise RuntimeError("U03 did not reproduce its intended symptom")


def u04() -> dict[str, Any]:
    x = torch.randn(64, 257, device="cuda")

    def measure(sync_each_call: bool) -> float:
        samples = []
        for _ in range(40):
            start = time.perf_counter()
            ops.softmax_warp_reduce(x)
            if sync_each_call:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        torch.cuda.synchronize()
        return statistics.median(samples)

    baseline = measure(False)
    blocked = measure(True)
    return {
        "symptom": "CPU-side per-call latency increased while the kernel implementation was unchanged",
        "baseline_cpu_submission_median_us": baseline,
        "observed_cpu_call_median_us": blocked,
        "ratio": blocked / baseline if baseline else None,
        "next_tool": "NSYS",
    }


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "U01": u01,
    "U02": u02,
    "U03": u03,
    "U04": u04,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["random", *CASES], default="random")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--reveal", action="store_true")
    args = parser.parse_args()
    if args.reveal:
        selected = sorted(CASES) if args.case == "random" else [args.case]
        for case in selected:
            print(f"{case}: {ANSWERS[case]}")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected = random.Random(args.seed).choice(sorted(CASES)) if args.case == "random" else args.case
    result = {"case_id": selected, **CASES[selected]()}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
