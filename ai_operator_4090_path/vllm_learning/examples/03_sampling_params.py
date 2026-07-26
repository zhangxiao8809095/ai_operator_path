#!/usr/bin/env python3
"""Lesson 3: compare common SamplingParams on the same prompt."""

from __future__ import annotations

import argparse

from vllm_lab import LabConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="给一个正在学习 CUDA 和大模型推理的工程师设计一句鼓励语。",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig.from_env()

    from vllm import LLM, SamplingParams

    llm = LLM(**config.llm_kwargs())
    experiments = [
        (
            "greedy",
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        ),
        (
            "balanced",
            SamplingParams(
                temperature=0.7,
                top_p=0.9,
                seed=config.seed,
                max_tokens=args.max_tokens,
            ),
        ),
        (
            "creative",
            SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=50,
                seed=config.seed,
                max_tokens=args.max_tokens,
            ),
        ),
    ]

    print(f"Prompt: {args.prompt}")
    for name, sampling in experiments:
        text = llm.generate([args.prompt], sampling)[0].outputs[0].text.strip()
        print(f"\n--- {name} ---\n{text}")


if __name__ == "__main__":
    main()
