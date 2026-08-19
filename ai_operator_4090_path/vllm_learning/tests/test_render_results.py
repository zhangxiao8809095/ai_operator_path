from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.render_results import render_document


class RenderResultsTests(unittest.TestCase):
    def test_missing_reports_are_explicit_and_paths_are_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = render_document(Path(directory))
        self.assertEqual(document.count("| 未运行"), 10)
        self.assertIn("reports/experiments/00_preflight.json", document)
        self.assertNotIn(directory, document)

    def test_profile_and_prefix_reports_are_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            chunked = {
                "profile": "chunked-512",
                "status": "completed",
                "server_configuration": {
                    "enable_chunked_prefill": "true",
                    "max_num_batched_tokens": "512",
                },
                "decode_request": {"status": "completed", "ttft_seconds": 0.1},
                "prefill_request": {
                    "status": "completed",
                    "prompt_tokens": 2048,
                    "ttft_seconds": 0.2,
                },
                "decode_itl_seconds": {
                    "before": {"p50": 0.01},
                    "during": {"p50": 0.02},
                    "after": {"p50": 0.01},
                },
            }
            prefix = {
                "status": "completed",
                "summary": {
                    "shared_cached": {"count": 3, "p50": 0.1, "p95": 0.11},
                    "unseen_control": {"count": 3, "p50": 0.2, "p95": 0.21},
                },
            }
            (report_dir / "08_chunked_prefill_chunked-512.json").write_text(
                json.dumps(chunked),
                encoding="utf-8",
            )
            (report_dir / "09_prefix_caching.json").write_text(
                json.dumps(prefix),
                encoding="utf-8",
            )
            document = render_document(report_dir)

        self.assertIn("chunked-512", document)
        self.assertIn("shared_cached", document)
        self.assertEqual(document.count("| 未运行"), 8)


if __name__ == "__main__":
    unittest.main()
