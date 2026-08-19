#!/usr/bin/env python3
"""Experiment 04: compare sampling controls, reproducibility and stop reasons."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import completion_record, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="为学习vLLM的工程师写一句技术风格的鼓励语。",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/04_sampling_diagnostics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens < 1 or args.repetitions < 1:
        raise SystemExit("max-tokens and repetitions must be positive")
    config = LabConfig.from_env()

    from vllm import LLM, SamplingParams

    llm = LLM(**config.llm_kwargs())
    settings = [
        ("greedy", SamplingParams(temperature=0.0, max_tokens=args.max_tokens)),
        (
            "low_temperature",
            SamplingParams(
                temperature=0.2,
                top_p=0.9,
                seed=config.seed,
                max_tokens=args.max_tokens,
            ),
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
            "top_k_limited",
            SamplingParams(
                temperature=0.7,
                top_p=1.0,
                top_k=10,
                seed=config.seed,
                max_tokens=args.max_tokens,
            ),
        ),
        (
            "forced_length",
            SamplingParams(
                temperature=0.0,
                max_tokens=8,
                ignore_eos=True,
            ),
        ),
    ]

    results = []
    for name, params in settings:
        runs = []
        for repetition in range(args.repetitions):
            output = llm.generate([args.prompt], params)[0]
            runs.append({"repetition": repetition + 1, **completion_record(output)})
        results.append(
            {
                "setting": name,
                "sampling_params": str(params),
                "unique_token_sequences": len({tuple(row["token_ids"]) for row in runs}),
                "runs": runs,
            }
        )

    report = {
        "experiment_id": "04_sampling_diagnostics",
        "title": "采样参数、复现性与停止原因",
        "status": "completed",
        "config": config.llm_kwargs(),
        "prompt": args.prompt,
        "results": results,
    }
    output_path = write_json(args.output, report)
    for setting in results:
        print(
            f"{setting['setting']}: unique={setting['unique_token_sequences']}, "
            f"finish={setting['runs'][0]['finish_reason']}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
