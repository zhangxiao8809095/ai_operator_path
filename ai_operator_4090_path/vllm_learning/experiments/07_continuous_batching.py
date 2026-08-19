#!/usr/bin/env python3
"""Experiment 07: stagger long/short streaming requests and sample scheduler metrics."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from vllm_lab.experiment_utils import (
    http_get_text,
    parse_prometheus_metrics,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-requests", type=int, default=2)
    parser.add_argument("--short-requests", type=int, default=4)
    parser.add_argument("--long-max-tokens", type=int, default=256)
    parser.add_argument("--short-max-tokens", type=int, default=32)
    parser.add_argument("--arrival-delay", type=float, default=0.2)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/07_continuous_batching.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numeric = [
        args.long_requests,
        args.short_requests,
        args.long_max_tokens,
        args.short_max_tokens,
    ]
    if min(numeric) < 1 or min(args.arrival_delay, args.metrics_interval, args.timeout) <= 0:
        raise SystemExit("request counts/tokens and timing arguments must be positive")

    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "local-token")
    served_model = os.environ.get("VLLM_SERVED_MODEL_NAME", "vllm-lab")
    root_url = f"http://{host}:{port}"
    base_url = f"{root_url}/v1"
    experiment_start = perf_counter()
    stop_metrics = threading.Event()
    metrics_samples: list[dict[str, Any]] = []
    metrics_errors: list[str] = []

    def relative_time() -> float:
        return perf_counter() - experiment_start

    def poll_metrics() -> None:
        while not stop_metrics.is_set():
            try:
                text = http_get_text(f"{root_url}/metrics", timeout=5.0)
                metrics_samples.append(
                    {
                        "time": relative_time(),
                        **parse_prometheus_metrics(text, METRIC_NAMES),
                    }
                )
            except Exception as exc:
                metrics_errors.append(f"{type(exc).__name__}: {exc}")
            stop_metrics.wait(args.metrics_interval)

    def run_request(kind: str, index: int, max_tokens: int) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)
        arrived = relative_time()
        prompt = (
            "请持续输出关于vLLM调度、KV cache和PagedAttention的技术要点。"
            if kind == "long"
            else "一句话解释continuous batching。"
        )
        first_token: float | None = None
        finish_reason: str | None = None
        text_parts: list[str] = []
        try:
            stream = client.completions.create(
                model=served_model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"ignore_eos": True},
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.text:
                    if first_token is None:
                        first_token = relative_time()
                    text_parts.append(choice.text)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
            finished = relative_time()
            return {
                "request": f"{kind}-{index}",
                "kind": kind,
                "status": "completed",
                "arrival_time": arrived,
                "first_token_time": first_token,
                "finish_time": finished,
                "ttft_seconds": None if first_token is None else first_token - arrived,
                "e2e_seconds": finished - arrived,
                "finish_reason": finish_reason,
                "characters": len("".join(text_parts)),
            }
        except Exception as exc:
            return {
                "request": f"{kind}-{index}",
                "kind": kind,
                "status": "failed",
                "arrival_time": arrived,
                "finish_time": relative_time(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    metrics_thread = threading.Thread(target=poll_metrics, daemon=True)
    metrics_thread.start()
    results = []
    workers = args.long_requests + args.short_requests
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_request, "long", index, args.long_max_tokens)
            for index in range(args.long_requests)
        ]
        time.sleep(args.arrival_delay)
        futures.extend(
            executor.submit(run_request, "short", index, args.short_max_tokens)
            for index in range(args.short_requests)
        )
        for future in as_completed(futures):
            results.append(future.result())
    stop_metrics.set()
    metrics_thread.join(timeout=max(2.0, args.metrics_interval * 4))
    results.sort(key=lambda item: item["arrival_time"])

    long_finished = [
        item["finish_time"]
        for item in results
        if item["kind"] == "long" and item["status"] == "completed"
    ]
    short_first = [
        item["first_token_time"]
        for item in results
        if item["kind"] == "short"
        and item["status"] == "completed"
        and item["first_token_time"] is not None
    ]
    dynamic_join_evidence = bool(
        long_finished and short_first and min(short_first) < max(long_finished)
    )
    completed = all(item["status"] == "completed" for item in results)
    report = {
        "experiment_id": "07_continuous_batching",
        "title": "错峰到达Continuous Batching与Metrics时间线",
        "status": "completed" if completed else "failed",
        "service": {"root_url": root_url, "served_model": served_model},
        "workload": vars(args) | {"output": str(args.output)},
        "dynamic_join_evidence": dynamic_join_evidence,
        "results": results,
        "metrics_samples": metrics_samples,
        "metrics_errors": sorted(set(metrics_errors)),
    }
    output_path = write_json(args.output, report)
    for item in results:
        print(
            f"{item['request']:8s} {item['status']:9s} "
            f"arrival={item['arrival_time']:.3f} finish={item['finish_time']:.3f}"
        )
    print(f"Dynamic-join evidence: {dynamic_join_evidence}")
    print(f"Report: {output_path}")
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
