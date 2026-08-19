#!/usr/bin/env python3
"""Experiment 01: separate engine initialization, warm-up and steady-state latency."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import completion_record, summarize, timed_call, write_json
from vllm_lab.gpu import take_gpu_snapshots

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="解释Prefill、Decode和KV cache之间的关系。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/01_engine_lifecycle.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 2 or args.max_tokens < 1:
        raise SystemExit("repetitions must be >= 2 and max-tokens must be positive")
    config = LabConfig.from_env()
    engine_kwargs = config.llm_kwargs() | {"enable_prefix_caching": False}

    import torch
    from vllm import LLM, SamplingParams

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    before = [item.__dict__ for item in take_gpu_snapshots()]
    llm, initialization_seconds = timed_call(
        lambda: LLM(**engine_kwargs),
        synchronize,
    )
    after_load = [item.__dict__ for item in take_gpu_snapshots()]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    warm_output, warmup_seconds = timed_call(
        lambda: llm.generate([args.prompt], sampling)[0],
        synchronize,
    )
    runs = []
    for run_index in range(args.repetitions):
        output, elapsed = timed_call(
            lambda: llm.generate([args.prompt], sampling)[0],
            synchronize,
        )
        runs.append(
            {
                "run": run_index + 1,
                "seconds": elapsed,
                **completion_record(output),
            }
        )

    report = {
        "experiment_id": "01_engine_lifecycle",
        "title": "引擎初始化、预热与稳态延迟",
        "status": "completed",
        "config": engine_kwargs,
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_output": completion_record(warm_output),
        "steady_state_seconds": summarize([item["seconds"] for item in runs]),
        "gpu_before": before,
        "gpu_after_load": after_load,
        "results": runs,
    }
    output_path = write_json(args.output, report)
    print(f"Initialization: {initialization_seconds:.3f}s")
    print(f"Warm-up:       {warmup_seconds:.3f}s")
    print(f"Steady P50:    {report['steady_state_seconds']['p50']:.3f}s")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
