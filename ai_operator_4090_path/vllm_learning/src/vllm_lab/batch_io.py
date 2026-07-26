"""JSONL input/output helpers used by the offline batch lesson."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    """A validated prompt read from a JSONL file."""

    id: str
    prompt: str


def load_prompts(path: str | Path) -> list[PromptRecord]:
    """Load records containing non-empty ``id`` and ``prompt`` fields."""

    input_path = Path(path)
    records: list[PromptRecord] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")

            record_id = payload.get("id", str(line_number))
            prompt = payload.get("prompt")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{input_path}:{line_number}: 'id' must be a non-empty string")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{input_path}:{line_number}: 'prompt' must be a non-empty string")
            records.append(PromptRecord(id=record_id, prompt=prompt))

    if not records:
        raise ValueError(f"{input_path}: no prompts found")
    return records


def write_results(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON-serializable result rows, creating the parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def combine_results(
    records: Sequence[PromptRecord],
    generated_texts: Sequence[str],
) -> list[dict[str, str]]:
    """Pair model outputs with their source records while checking cardinality."""

    if len(records) != len(generated_texts):
        raise ValueError(
            f"record/output count mismatch: {len(records)} records, "
            f"{len(generated_texts)} outputs"
        )
    return [
        {"id": record.id, "prompt": record.prompt, "generated_text": text}
        for record, text in zip(records, generated_texts, strict=True)
    ]
