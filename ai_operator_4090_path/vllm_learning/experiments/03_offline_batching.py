#!/usr/bin/env python3
"""Experiment 03: compare list batching with synchronous per-request generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import (
    build_prompt_near_tokens,
    completion_record,
    summarize,
    timed_call,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/03_offline_batching.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.requests, args.prompt_tokens, args.max_tokens, args.repetitions) < 1:
        raise SystemExit("all numeric arguments must be positive")
    config = LabConfig.from_env()
    engine_kwargs = config.llm_kwargs() | {"enable_prefix_caching": False}
    if args.prompt_tokens + args.max_tokens > config.max_model_len:
        raise SystemExit("prompt-tokens + max-tokens exceeds max_model_len")

    import torch
    from vllm import LLM, SamplingParams

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    base_prompt, actual_prompt_tokens = build_prompt_near_tokens(
        tokenizer,
        args.prompt_tokens,
        "离线批量实验使用固定上下文来控制输入形状。",
    )
    prompts = [base_prompt] * args.requests
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    llm.generate(["预热。"], SamplingParams(temperature=0.0, max_tokens=8))

    rows = []
    for repetition in range(args.repetitions):
        modes = ("list_batch", "sync_single") if repetition % 2 == 0 else (
            "sync_single",
            "list_batch",
        )
        for mode in modes:
            if mode == "list_batch":
                outputs, elapsed = timed_call(
                    lambda: llm.generate(prompts, sampling),
                    synchronize,
                )
            else:
                outputs, elapsed = timed_call(
                    lambda: [llm.generate([prompt], sampling)[0] for prompt in prompts],
                    synchronize,
                )
            records = [completion_record(output) for output in outputs]
            output_tokens = sum(record["output_tokens"] for record in records)
            rows.append(
                {
                    "repetition": repetition + 1,
                    "mode": mode,
                    "seconds": elapsed,
                    "requests": len(outputs),
                    "output_tokens": output_tokens,
                    "output_tokens_per_second": output_tokens / elapsed,
                }
            )

    summary = {}
    for mode in ("list_batch", "sync_single"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "seconds": summarize([row["seconds"] for row in selected]),
            "output_tokens_per_second": summarize(
                [row["output_tokens_per_second"] for row in selected]
            ),
        }
    report = {
        "experiment_id": "03_offline_batching",
        "title": "离线列表批量与同步逐条调用对照",
        "status": "completed",
        "config": engine_kwargs,
        "workload": {
            "requests": args.requests,
            "target_prompt_tokens": args.prompt_tokens,
            "base_prompt_tokens": actual_prompt_tokens,
            "max_tokens": args.max_tokens,
        },
        "summary": summary,
        "results": rows,
    }
    output_path = write_json(args.output, report)
    for mode, values in summary.items():
        print(
            f"{mode:12s}: p50={values['seconds']['p50']:.3f}s, "
            f"tokens/s={values['output_tokens_per_second']['p50']:.2f}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
