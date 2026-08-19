from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vllm_lab.experiment_utils import (
    display_width,
    markdown_table,
    parse_int_csv,
    parse_prometheus_metrics,
    percentile,
    summarize,
    write_json,
)


class ExperimentUtilsTests(unittest.TestCase):
    def test_parse_int_csv(self) -> None:
        self.assertEqual(parse_int_csv("1, 8,16", "sizes"), [1, 8, 16])
        with self.assertRaisesRegex(ValueError, "sizes"):
            parse_int_csv("1,zero", "sizes")
        with self.assertRaisesRegex(ValueError, ">= 1"):
            parse_int_csv("0,1", "sizes")

    def test_percentiles_and_summary(self) -> None:
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        result = summarize([1, 2, 3, 4])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["p50"], 2.5)

    def test_parse_prometheus_metrics(self) -> None:
        text = """
# HELP ignored something
vllm:num_requests_running{model_name="lab"} 2
vllm:num_requests_running{model_name="lab-2"} 3
vllm:kv_cache_usage_perc 0.75
"""
        parsed = parse_prometheus_metrics(
            text,
            ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"],
        )
        self.assertEqual(parsed["vllm:num_requests_running"], 5)
        self.assertEqual(parsed["vllm:kv_cache_usage_perc"], 0.75)

    def test_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/report.json"
            self.assertEqual(write_json(path, {"中文": 1}), path)
            self.assertIn("中文", path.read_text(encoding="utf-8"))

    def test_markdown_table_aligns_cjk_source_columns(self) -> None:
        table = markdown_table(
            ["概念", "value"],
            [["连续批处理", 1], ["KV cache", 20]],
        )
        lines = table.splitlines()
        pipe_positions = [
            [display_width(line[:index]) for index, char in enumerate(line) if char == "|"]
            for line in lines
        ]
        self.assertTrue(all(positions == pipe_positions[0] for positions in pipe_positions))

    def test_markdown_table_escapes_pipes(self) -> None:
        table = markdown_table(["name"], [["a|b"]])
        self.assertIn(r"a\|b", table)
        with self.assertRaisesRegex(ValueError, "same number"):
            markdown_table(["a", "b"], [["only-one"]])


if __name__ == "__main__":
    unittest.main()
