#!/usr/bin/env python3
"""Call the local vLLM server with the official OpenAI Python client."""

from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    host = os.environ.get("VLLM_HOST", "127.0.0.1")
    port = os.environ.get("VLLM_PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "local-token")
    served_model = os.environ.get("VLLM_SERVED_MODEL_NAME", "vllm-lab")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=served_model,
        messages=[
            {"role": "system", "content": "你是一位简洁的 vLLM 教学助手。"},
            {"role": "user", "content": "一句话说明 PagedAttention 解决了什么问题。"},
        ],
        temperature=0.2,
        max_tokens=128,
        extra_body={"top_k": 40},
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
