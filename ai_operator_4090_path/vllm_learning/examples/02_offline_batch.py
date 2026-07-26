#!/usr/bin/env python3
"""Lesson 2: load JSONL prompts and run one offline batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lab import LabConfig
from vllm_lab.batch_io import combine_results, load_prompts, write_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/prompts.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/offline_results.jsonl",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig.from_env()
    records = load_prompts(args.input)

    from vllm import LLM, SamplingParams

    llm = LLM(**config.llm_kwargs())
    sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=args.max_tokens)
    outputs = llm.generate([record.prompt for record in records], sampling)
    rows = combine_results(records, [item.outputs[0].text.strip() for item in outputs])
    write_results(args.output, rows)

    print(f"Wrote {len(rows)} results to {args.output}")
    for row in rows:
        print(f"\n[{row['id']}] {row['generated_text']}")


if __name__ == "__main__":
    main()
