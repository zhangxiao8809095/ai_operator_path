from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vllm_lab.batch_io import combine_results, load_prompts, write_results


class BatchIoTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "prompts.jsonl"
            input_path.write_text(
                '{"id": "a", "prompt": "hello"}\n{"id": "b", "prompt": "world"}\n',
                encoding="utf-8",
            )

            records = load_prompts(input_path)
            rows = combine_results(records, ["first", "second"])
            output_path = root / "nested/results.jsonl"
            write_results(output_path, rows)

            decoded = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record.id for record in records], ["a", "b"])
            self.assertEqual(decoded, rows)

    def test_rejects_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"id": "a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "'prompt'"):
                load_prompts(path)

    def test_rejects_result_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.jsonl"
            path.write_text('{"id": "a", "prompt": "hello"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                combine_results(load_prompts(path), [])


if __name__ == "__main__":
    unittest.main()
