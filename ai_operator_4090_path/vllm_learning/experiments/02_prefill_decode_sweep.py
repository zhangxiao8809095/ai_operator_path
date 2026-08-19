#!/usr/bin/env python3
"""Experiment 02: sweep prompt/output lengths to separate Prefill and Decode effects."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.experiment_utils import (
    build_prompt_near_tokens,
    completion_record,
    parse_int_csv,
    summarize,
    timed_call,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--output-tokens", default="16,64,256")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/02_prefill_decode_sweep.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_targets = parse_int_csv(args.prompt_tokens, "prompt-tokens")
    output_targets = parse_int_csv(args.output_tokens, "output-tokens")
    if args.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    config = LabConfig.from_env()
    engine_kwargs = config.llm_kwargs() | {"enable_prefix_caching": False}
    for prompt_tokens in prompt_targets:
        for output_tokens in output_targets:
            if prompt_tokens + output_tokens > config.max_model_len:
                raise SystemExit(
                    f"case {prompt_tokens}+{output_tokens} exceeds max_model_len="
                    f"{config.max_model_len}"
                )

    import torch
    from vllm import LLM, SamplingParams

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    llm.generate(["预热。"], SamplingParams(temperature=0.0, max_tokens=8))
    prompts = {
        target: build_prompt_near_tokens(
            tokenizer,
            target,
            "这是一段用于区分Prefill和Decode开销的固定上下文。",
        )
        for target in prompt_targets
    }

    cases = []
    for prompt_target in prompt_targets:
        prompt, built_tokens = prompts[prompt_target]
        for output_target in output_targets:
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=output_target,
                ignore_eos=True,
            )
            runs = []
            for repetition in range(args.repetitions):
                result, elapsed = timed_call(
                    lambda: llm.generate([prompt], sampling)[0],
                    synchronize,
                )
                row = completion_record(result)
                row.update({"repetition": repetition + 1, "seconds": elapsed})
                runs.append(row)
            cases.append(
                {
                    "prompt_target": prompt_target,
                    "built_prompt_tokens": built_tokens,
                    "output_target": output_target,
                    "latency_seconds": summarize([row["seconds"] for row in runs]),
                    "results": runs,
                }
            )

    report = {
        "experiment_id": "02_prefill_decode_sweep",
        "title": "Prompt/Output长度二维扫描",
        "status": "completed",
        "config": engine_kwargs,
        "results": cases,
        "note": "Offline total latency is not a direct TTFT/TPOT measurement.",
    }
    output_path = write_json(args.output, report)
    for case in cases:
        print(
            f"prompt={case['built_prompt_tokens']:4d}, output={case['output_target']:3d}, "
            f"p50={case['latency_seconds']['p50']:.3f}s"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
