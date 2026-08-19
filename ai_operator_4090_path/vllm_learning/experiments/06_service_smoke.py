#!/usr/bin/env python3
"""Experiment 06: validate service health, auth, model routing, completions and chat."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vllm_lab.experiment_utils import http_get_text, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/06_service_smoke.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "local-token")
    served_model = os.environ.get("VLLM_SERVED_MODEL_NAME", "vllm-lab")
    root_url = f"http://{host}:{port}"
    checks: list[dict[str, Any]] = []

    try:
        health = http_get_text(f"{root_url}/health", timeout=args.timeout)
        checks.append({"name": "health", "passed": True, "detail": health.strip()})

        models_text = http_get_text(
            f"{root_url}/v1/models",
            api_key,
            timeout=args.timeout,
        )
        model_payload = json.loads(models_text)
        model_ids = [item["id"] for item in model_payload.get("data", [])]
        checks.append(
            {
                "name": "served_model_name",
                "passed": served_model in model_ids,
                "detail": model_ids,
            }
        )

        from openai import OpenAI

        client = OpenAI(
            base_url=f"{root_url}/v1",
            api_key=api_key,
            timeout=args.timeout,
        )
        completion = client.completions.create(
            model=served_model,
            prompt="用一句话解释KV cache。",
            temperature=0.0,
            max_tokens=32,
        )
        checks.append(
            {
                "name": "completions",
                "passed": bool(completion.choices[0].text),
                "detail": completion.choices[0].text.strip(),
            }
        )
        chat = client.chat.completions.create(
            model=served_model,
            messages=[{"role": "user", "content": "用一句话解释PagedAttention。"}],
            temperature=0.0,
            max_tokens=32,
        )
        chat_text = chat.choices[0].message.content or ""
        checks.append(
            {
                "name": "chat_completions",
                "passed": bool(chat_text),
                "detail": chat_text.strip(),
            }
        )
        metrics = http_get_text(f"{root_url}/metrics", timeout=args.timeout)
        checks.append(
            {
                "name": "metrics",
                "passed": "vllm:" in metrics,
                "detail": f"{len(metrics.splitlines())} lines",
            }
        )
    except Exception as exc:  # server/client exceptions must be captured in the report
        checks.append(
            {
                "name": "uncaught_service_layer",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    passed = bool(checks) and all(item["passed"] for item in checks)
    report = {
        "experiment_id": "06_service_smoke",
        "title": "OpenAI兼容服务分层验收",
        "status": "passed" if passed else "failed",
        "service": {"root_url": root_url, "served_model": served_model},
        "checks": checks,
    }
    output_path = write_json(args.output, report)
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}")
    print(f"Report: {output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
