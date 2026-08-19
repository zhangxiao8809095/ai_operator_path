#!/usr/bin/env python3
"""Experiment 09: compare cached shared prefixes with equal-length unseen prefixes."""

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
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/09_prefix_caching.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.prefix_tokens, args.max_tokens, args.repetitions) < 1:
        raise SystemExit("numeric arguments must be positive")
    config = LabConfig.from_env()
    if args.prefix_tokens + args.max_tokens + 32 > config.max_model_len:
        raise SystemExit("prefix and output leave insufficient room under max_model_len")

    import torch
    from vllm import LLM, SamplingParams

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    kwargs = config.llm_kwargs() | {"enable_prefix_caching": True}
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    shared_prefix, shared_tokens = build_prompt_near_tokens(
        tokenizer,
        args.prefix_tokens,
        "共享技术文档描述了调度器、KV块和PagedAttention的数据流。",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    llm.generate([shared_prefix + "\n问题：概括调度器。"], sampling)
    rows = []
    for repetition in range(args.repetitions):
        unseen_prefix, unseen_tokens = build_prompt_near_tokens(
            tokenizer,
            args.prefix_tokens,
            f"未见过的对照文档{repetition}讨论另一组完全不同的系统事实。",
        )
        cases = [
            ("shared_cached", shared_prefix, shared_tokens),
            ("unseen_control", unseen_prefix, unseen_tokens),
        ]
        if repetition % 2:
            cases.reverse()
        for case, prefix, actual_tokens in cases:
            output, elapsed = timed_call(
                lambda prefix=prefix, repetition=repetition: llm.generate(
                    [prefix + f"\n问题{repetition}：只回答已读取。"],
                    sampling,
                )[0],
                synchronize,
            )
            rows.append(
                {
                    "repetition": repetition + 1,
                    "case": case,
                    "prefix_tokens": actual_tokens,
                    "seconds": elapsed,
                    **completion_record(output),
                }
            )

    summary = {
        case: summarize([row["seconds"] for row in rows if row["case"] == case])
        for case in ("shared_cached", "unseen_control")
    }
    report = {
        "experiment_id": "09_prefix_caching",
        "title": "Automatic Prefix Caching共享前缀对照",
        "status": "completed",
        "config": kwargs,
        "summary": summary,
        "results": rows,
        "note": "Offline total latency is supporting evidence, not a direct server TTFT metric.",
    }
    output_path = write_json(args.output, report)
    for case, values in summary.items():
        print(f"{case:14s}: p50={values['p50']:.3f}s")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
