#!/usr/bin/env python3
"""Lesson 1: the shortest offline vLLM inference path."""

from __future__ import annotations

import argparse

from vllm_lab import LabConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="用三句话解释 vLLM 为什么能提高大语言模型推理吞吐量。",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig.from_env()

    from vllm import LLM, SamplingParams

    print(f"Loading {config.model!r} with tensor_parallel_size={config.tensor_parallel_size}")
    llm = LLM(**config.llm_kwargs())
    sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=args.max_tokens)
    output = llm.generate([args.prompt], sampling)[0]

    print(f"\nPrompt:\n{output.prompt}")
    print(f"\nGenerated:\n{output.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
