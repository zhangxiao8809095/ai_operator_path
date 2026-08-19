#!/usr/bin/env python3
"""Experiment 08: inject a long Prefill while another request is decoding."""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from vllm_lab.experiment_utils import summarize, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Label for this server configuration")
    parser.add_argument("--decode-max-tokens", type=int, default=512)
    parser.add_argument("--prefill-repetitions", type=int, default=120)
    parser.add_argument("--prefill-max-tokens", type=int, default=16)
    parser.add_argument("--injection-delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _summary_or_empty(values: list[float]) -> dict[str, float | int]:
    return summarize(values) if values else {}


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.profile) is None:
        raise SystemExit("profile may contain only letters, digits, dot, underscore, and hyphen")
    if min(
        args.decode_max_tokens,
        args.prefill_repetitions,
        args.prefill_max_tokens,
    ) < 1:
        raise SystemExit("token/repetition arguments must be positive")
    if min(args.injection_delay, args.timeout) <= 0:
        raise SystemExit("timing arguments must be positive")

    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "local-token")
    served_model = os.environ.get("VLLM_SERVED_MODEL_NAME", "vllm-lab")
    root_url = f"http://{host}:{port}"
    started = perf_counter()

    def now() -> float:
        return perf_counter() - started

    def stream_request(kind: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=f"{root_url}/v1", api_key=api_key, timeout=args.timeout)
        arrival = now()
        token_times: list[float] = []
        prompt_tokens: int | None = None
        finish_reason: str | None = None
        try:
            stream = client.completions.create(
                model=served_model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"ignore_eos": True},
            )
            for chunk in stream:
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.text:
                    token_times.append(now())
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
            finished = now()
            return {
                "kind": kind,
                "status": "completed",
                "arrival_time": arrival,
                "first_token_time": token_times[0] if token_times else None,
                "finish_time": finished,
                "ttft_seconds": None if not token_times else token_times[0] - arrival,
                "e2e_seconds": finished - arrival,
                "prompt_tokens": prompt_tokens,
                "stream_events": len(token_times),
                "token_times": token_times,
                "finish_reason": finish_reason,
            }
        except Exception as exc:
            return {
                "kind": kind,
                "status": "failed",
                "arrival_time": arrival,
                "finish_time": now(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    decode_prompt = "持续输出vLLM调度、KV cache和PagedAttention的技术要点。"
    prefill_prompt = (
        "这是用于制造长Prefill的固定技术上下文。请记住其中每一句。"
        * args.prefill_repetitions
        + "\n只回答：上下文已读取。"
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        decode_future = executor.submit(
            stream_request,
            "existing_decode",
            decode_prompt,
            args.decode_max_tokens,
        )
        time.sleep(args.injection_delay)
        prefill_future = executor.submit(
            stream_request,
            "injected_prefill",
            prefill_prompt,
            args.prefill_max_tokens,
        )
        prefill_result = prefill_future.result()
        decode_result = decode_future.result()

    segment_itl: dict[str, dict[str, float | int]] = {}
    if decode_result["status"] == "completed" and prefill_result["status"] == "completed":
        token_times = decode_result["token_times"]
        segments: dict[str, list[float]] = {"before": [], "during": [], "after": []}
        for previous, current in zip(token_times, token_times[1:], strict=False):
            if current < prefill_result["arrival_time"]:
                segment = "before"
            elif current <= prefill_result["finish_time"]:
                segment = "during"
            else:
                segment = "after"
            segments[segment].append(current - previous)
        segment_itl = {name: _summary_or_empty(values) for name, values in segments.items()}

    requests_completed = all(
        item["status"] == "completed" for item in (decode_result, prefill_result)
    )
    overlap_evidence = bool(
        requests_completed
        and decode_result["arrival_time"] < prefill_result["arrival_time"]
        and prefill_result["arrival_time"] < decode_result["finish_time"]
        and segment_itl.get("during", {}).get("count", 0)
    )
    output_path = args.output or (
        PROJECT_ROOT / f"reports/experiments/08_chunked_prefill_{args.profile}.json"
    )
    report = {
        "experiment_id": "08_chunked_prefill",
        "title": "长Prefill注入与Decode ITL干扰",
        "profile": args.profile,
        "status": (
            "completed"
            if overlap_evidence
            else "inconclusive"
            if requests_completed
            else "failed"
        ),
        "service": {"root_url": root_url, "served_model": served_model},
        "server_configuration": {
            "enable_chunked_prefill": os.environ.get(
                "VLLM_ENABLE_CHUNKED_PREFILL",
                "not-recorded",
            ),
            "max_num_batched_tokens": os.environ.get(
                "VLLM_MAX_NUM_BATCHED_TOKENS",
                "not-recorded",
            ),
        },
        "workload": {
            "decode_max_tokens": args.decode_max_tokens,
            "prefill_repetitions": args.prefill_repetitions,
            "prefill_max_tokens": args.prefill_max_tokens,
            "injection_delay": args.injection_delay,
        },
        "decode_request": decode_result,
        "prefill_request": prefill_result,
        "decode_itl_seconds": segment_itl,
        "overlap_evidence": overlap_evidence,
        "note": "Streaming event intervals approximate client-observed ITL and include transport.",
    }
    saved = write_json(output_path, report)
    print(f"Profile: {args.profile}")
    print(f"Status: {report['status']}")
    print(f"Report: {saved}")
    if not overlap_evidence:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
