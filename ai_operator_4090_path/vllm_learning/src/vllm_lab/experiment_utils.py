"""GPU-independent helpers shared by the debugging experiments."""

from __future__ import annotations

import json
import math
import re
import unicodedata
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


def parse_int_csv(value: str, variable: str, *, minimum: int = 1) -> list[int]:
    """Parse a comma-separated integer list while preserving input order."""

    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{variable} must contain comma-separated integers") from exc
    if not values:
        raise ValueError(f"{variable} must not be empty")
    if any(item < minimum for item in values):
        raise ValueError(f"{variable} values must be >= {minimum}")
    return values


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Return a compact latency/throughput summary."""

    if not values:
        raise ValueError("summarize requires at least one value")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": mean(numeric),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "max": max(numeric),
    }


def timed_call(
    function: Callable[[], T],
    synchronize: Callable[[], None] | None = None,
) -> tuple[T, float]:
    """Measure a synchronous operation, optionally synchronizing the device."""

    if synchronize is not None:
        synchronize()
    started = perf_counter()
    result = function()
    if synchronize is not None:
        synchronize()
    return result, perf_counter() - started


def write_json(path: str | Path, payload: Any) -> Path:
    """Write an indented UTF-8 JSON report and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_prompt_near_tokens(tokenizer: Any, target_tokens: int, seed_text: str) -> tuple[str, int]:
    """Build text whose encoded length is close to, and no larger than, a target."""

    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    seed = seed_text.strip()
    if not seed:
        raise ValueError("seed_text must not be empty")

    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("seed_text produced no token IDs")
    repeats = max(1, math.ceil(target_tokens / len(seed_ids)) + 1)
    expanded = (seed + " ") * repeats
    expanded_ids = tokenizer.encode(expanded, add_special_tokens=False)
    prompt = tokenizer.decode(expanded_ids[:target_tokens], skip_special_tokens=True)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_tokens > target_tokens:
        raise RuntimeError("prompt builder exceeded its target token count")
    return prompt, actual_tokens


def completion_record(request_output: Any) -> dict[str, Any]:
    """Convert the first vLLM completion to JSON-friendly diagnostics."""

    candidate = request_output.outputs[0]
    prompt_ids = getattr(request_output, "prompt_token_ids", None) or []
    token_ids = list(candidate.token_ids)
    return {
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(token_ids),
        "finish_reason": candidate.finish_reason,
        "stop_reason": getattr(candidate, "stop_reason", None),
        "token_ids": token_ids,
        "text": candidate.text.strip(),
    }


def http_get_text(url: str, api_key: str | None = None, *, timeout: float = 10.0) -> str:
    """Fetch a UTF-8 HTTP endpoint with an optional bearer token."""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8")


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def parse_prometheus_metrics(text: str, names: Iterable[str]) -> dict[str, float]:
    """Sum selected Prometheus samples, which is appropriate for this single-GPU lab."""

    selected = set(names)
    result = {name: 0.0 for name in selected}
    found: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in selected:
            result[name] += float(match.group("value"))
            found.add(name)
    return {name: result[name] for name in selected if name in found}


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def display_width(value: str) -> int:
    """Return an approximate terminal display width for Markdown source alignment."""

    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def markdown_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    """Render a left-aligned Markdown table whose source columns align with CJK text."""

    normalized_headers = [_markdown_cell(value) for value in headers]
    normalized_rows = [[_markdown_cell(value) for value in row] for row in rows]
    if not normalized_headers:
        raise ValueError("headers must not be empty")
    if any(len(row) != len(normalized_headers) for row in normalized_rows):
        raise ValueError("every row must have the same number of cells as headers")

    widths = []
    for column, header in enumerate(normalized_headers):
        cell_widths = [display_width(row[column]) for row in normalized_rows]
        widths.append(max(3, display_width(header), *cell_widths))

    def render_row(cells: Sequence[str]) -> str:
        padded = [
            cell + " " * (widths[index] - display_width(cell))
            for index, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    separator = ["-" * width for width in widths]
    lines = [render_row(normalized_headers), render_row(separator)]
    lines.extend(render_row(row) for row in normalized_rows)
    return "\n".join(lines)
