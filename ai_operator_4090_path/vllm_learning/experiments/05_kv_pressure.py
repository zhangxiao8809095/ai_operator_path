#!/usr/bin/env python3
"""Experiment 05: increase active tokens safely and observe memory/latency pressure."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import (
    build_prompt_near_tokens,
    completion_record,
    parse_int_csv,
    timed_call,
    write_json,
)
from vllm_lab.gpu import take_gpu_snapshots

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,16")
    parser.add_argument("--prompt-tokens", default="512,2048")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/05_kv_pressure.json",
    )
    return parser.parse_args()


class GpuMonitor:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    {
                        "time": time.time(),
                        "gpus": [item.__dict__ for item in take_gpu_snapshots()],
                    }
                )
            except (OSError, RuntimeError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> GpuMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 4))


def main() -> None:
    args = parse_args()
    batch_sizes = parse_int_csv(args.batch_sizes, "batch-sizes")
    prompt_targets = parse_int_csv(args.prompt_tokens, "prompt-tokens")
    if args.max_tokens < 1 or args.sample_interval <= 0:
        raise SystemExit("max-tokens and sample-interval must be positive")
    config = LabConfig.from_env()
    engine_kwargs = config.llm_kwargs() | {"enable_prefix_caching": False}
    if max(prompt_targets) + args.max_tokens > config.max_model_len:
        raise SystemExit("largest prompt + max-tokens exceeds max_model_len")

    import torch
    from vllm import LLM, SamplingParams

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    before_load = [item.__dict__ for item in take_gpu_snapshots()]
    llm = LLM(**engine_kwargs)
    after_load = [item.__dict__ for item in take_gpu_snapshots()]
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    prompts = {
        target: build_prompt_near_tokens(
            tokenizer,
            target,
            "这是用于观察KV cache压力的固定重复上下文。",
        )
        for target in prompt_targets
    }
    scenarios = []
    for prompt_target in prompt_targets:
        prompt, actual_tokens = prompts[prompt_target]
        for batch_size in batch_sizes:
            scenario: dict[str, Any] = {
                "prompt_target": prompt_target,
                "actual_prompt_tokens": actual_tokens,
                "batch_size": batch_size,
                "potential_active_tokens": batch_size
                * (actual_tokens + args.max_tokens),
            }
            try:
                with GpuMonitor(args.sample_interval) as monitor:
                    outputs, elapsed = timed_call(
                        lambda: llm.generate([prompt] * batch_size, sampling),
                        synchronize,
                    )
                records = [completion_record(output) for output in outputs]
                scenario.update(
                    {
                        "status": "completed",
                        "seconds": elapsed,
                        "output_tokens": sum(row["output_tokens"] for row in records),
                        "gpu_samples": monitor.samples,
                    }
                )
            except (RuntimeError, ValueError) as exc:
                scenario.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                scenarios.append(scenario)
                if not args.continue_on_error:
                    report = {
                        "experiment_id": "05_kv_pressure",
                        "title": "KV容量与负载压力递增",
                        "status": "failed",
                        "config": engine_kwargs,
                        "gpu_before_load": before_load,
                        "gpu_after_load": after_load,
                        "results": scenarios,
                    }
                    write_json(args.output, report)
                    raise
                continue
            scenarios.append(scenario)
            print(
                f"batch={batch_size:2d}, prompt={actual_tokens:4d}, "
                f"seconds={scenario['seconds']:.3f}"
            )

    report = {
        "experiment_id": "05_kv_pressure",
        "title": "KV容量与负载压力递增",
        "status": "completed"
        if all(item["status"] == "completed" for item in scenarios)
        else "completed_with_failures",
        "config": engine_kwargs,
        "gpu_before_load": before_load,
        "gpu_after_load": after_load,
        "results": scenarios,
        "note": "nvidia-smi samples show process memory, not internal KV block usage.",
    }
    output_path = write_json(args.output, report)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
