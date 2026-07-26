#!/usr/bin/env python3
"""Lesson 4: observe vLLM's reserved memory around short and long requests."""

from __future__ import annotations

import argparse
import time

from vllm_lab import LabConfig
from vllm_lab.gpu import format_snapshots, take_gpu_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Keep the process alive after generation for external nvidia-smi observation.",
    )
    return parser.parse_args()


def show(label: str) -> None:
    print(format_snapshots(label, take_gpu_snapshots()), flush=True)


def main() -> None:
    args = parse_args()
    invalid_positive = args.batch_size < 1 or args.repeat < 1 or args.max_tokens < 1
    if invalid_positive or args.hold_seconds < 0:
        raise SystemExit(
            "batch-size/repeat/max-tokens must be positive; hold-seconds cannot be negative"
        )

    config = LabConfig.from_env()
    show("before model load")

    from vllm import LLM, SamplingParams

    llm = LLM(**config.llm_kwargs())
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    show("after model load and KV-cache reservation")

    llm.generate(["简要解释 KV cache。"], sampling)
    show("after one short request")

    long_prompt = (
        "下面是一段用于观察 KV cache 的重复上下文。请最后总结上下文的目的。"
        * args.repeat
    )
    llm.generate([long_prompt] * args.batch_size, sampling)
    show(f"after {args.batch_size} long requests")

    if args.hold_seconds:
        print(f"Holding process for {args.hold_seconds}s; inspect it with nvidia-smi.")
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
