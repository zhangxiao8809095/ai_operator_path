#!/usr/bin/env python3
"""Render experiment JSON reports into the independent Markdown result record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vllm_lab.experiment_utils import markdown_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports/experiments"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/vllm_experiment_results.md"
Report = dict[str, Any]
TableBuilder = Callable[[Report], tuple[list[str], list[list[object]], list[str]]]


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _detail(value: object, limit: int = 100) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _preflight(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = [
        [item["name"], "通过" if item["passed"] else "失败", _detail(item.get("actual"))]
        for item in report.get("checks", [])
    ]
    return ["检查项", "结果", "实际值"], rows, []


def _engine_lifecycle(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    steady = report.get("steady_state_seconds", {})
    rows = [
        [
            "引擎初始化",
            1,
            _number(report.get("initialization_seconds")),
            "含模型加载与KV规划",
        ],
        ["预热", 1, _number(report.get("warmup_seconds")), "首次generate"],
        ["稳态generate", steady.get("count", "N/A"), _number(steady.get("p50")), "P50"],
    ]
    return ["阶段", "样本数", "秒", "口径"], rows, []


def _prefill_decode(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = [
        [
            item.get("prompt_target"),
            item.get("built_prompt_tokens"),
            item.get("output_target"),
            _number(item.get("latency_seconds", {}).get("p50")),
            _number(item.get("latency_seconds", {}).get("p95")),
        ]
        for item in report.get("results", [])
    ]
    notes = [str(report["note"])] if report.get("note") else []
    headers = [
        "目标输入token",
        "实际输入token",
        "目标输出token",
        "总延迟P50(s)",
        "总延迟P95(s)",
    ]
    return headers, rows, notes


def _offline_batching(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = []
    for mode, item in report.get("summary", {}).items():
        rows.append(
            [
                mode,
                item.get("seconds", {}).get("count", "N/A"),
                _number(item.get("seconds", {}).get("p50")),
                _number(item.get("output_tokens_per_second", {}).get("p50"), 2),
            ]
        )
    return ["调用方式", "重复数", "总耗时P50(s)", "输出tokens/s P50"], rows, []


def _sampling(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = []
    for item in report.get("results", []):
        runs = item.get("runs", [])
        first = runs[0] if runs else {}
        rows.append(
            [
                item.get("setting"),
                len(runs),
                item.get("unique_token_sequences"),
                first.get("finish_reason", "N/A"),
                first.get("output_tokens", "N/A"),
                _detail(first.get("text", ""), 60),
            ]
        )
    headers = [
        "参数组",
        "重复数",
        "唯一token序列",
        "停止原因",
        "首轮输出token",
        "首轮输出摘录",
    ]
    return headers, rows, []


def _kv_pressure(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = [
        [
            item.get("batch_size"),
            item.get("prompt_target"),
            item.get("actual_prompt_tokens"),
            item.get("potential_active_tokens"),
            item.get("status"),
            _number(item.get("seconds")),
            _detail(item.get("error", ""), 60),
        ]
        for item in report.get("results", [])
    ]
    notes = [str(report["note"])] if report.get("note") else []
    headers = [
        "batch",
        "目标输入token",
        "实际输入token",
        "潜在活跃token",
        "状态",
        "秒",
        "错误",
    ]
    return headers, rows, notes


def _service_smoke(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = []
    for item in report.get("checks", []):
        rows.append(
            [
                item.get("name"),
                "通过" if item.get("passed") else "失败",
                _detail(item.get("detail")),
            ]
        )
    return ["服务层检查", "结果", "证据"], rows, []


def _continuous_batching(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = [
        [
            item.get("request"),
            item.get("kind"),
            item.get("status"),
            _number(item.get("arrival_time")),
            _number(item.get("ttft_seconds")),
            _number(item.get("e2e_seconds")),
            item.get("finish_reason", "N/A"),
            _detail(item.get("error", ""), 50),
        ]
        for item in report.get("results", [])
    ]
    evidence = "是" if report.get("dynamic_join_evidence") else "否"
    metrics_count = len(report.get("metrics_samples", []))
    notes = [f"短请求动态加入证据：{evidence}；metrics 采样数：{metrics_count}。"]
    return (
        ["请求", "类型", "状态", "到达(s)", "TTFT(s)", "E2E(s)", "停止原因", "错误"],
        rows,
        notes,
    )


def _chunked_prefill(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = []
    for profile in report.get("profiles", []):
        decode = profile.get("decode_request", {})
        prefill = profile.get("prefill_request", {})
        itl = profile.get("decode_itl_seconds", {})
        configuration = profile.get("server_configuration", {})
        rows.append(
            [
                profile.get("profile"),
                configuration.get("enable_chunked_prefill"),
                configuration.get("max_num_batched_tokens"),
                profile.get("status"),
                prefill.get("prompt_tokens", "N/A"),
                _number(prefill.get("ttft_seconds")),
                _number(decode.get("ttft_seconds")),
                _number(itl.get("before", {}).get("p50")),
                _number(itl.get("during", {}).get("p50")),
                _number(itl.get("after", {}).get("p50")),
            ]
        )
    headers = [
        "profile",
        "chunked",
        "token预算",
        "状态",
        "长Prompt token",
        "长Prompt TTFT(s)",
        "Decode TTFT(s)",
        "注入前ITL P50(s)",
        "注入中ITL P50(s)",
        "注入后ITL P50(s)",
    ]
    notes = ["客户端流事件间隔近似 ITL，包含本地传输和序列化开销。"]
    return headers, rows, notes


def _prefix_caching(report: Report) -> tuple[list[str], list[list[object]], list[str]]:
    rows = []
    for case, values in report.get("summary", {}).items():
        rows.append(
            [
                case,
                values.get("count", "N/A"),
                _number(values.get("p50")),
                _number(values.get("p95")),
            ]
        )
    notes = [str(report["note"])] if report.get("note") else []
    return ["前缀类型", "样本数", "总延迟P50(s)", "总延迟P95(s)"], rows, notes


EXPERIMENTS: list[tuple[str, str, TableBuilder]] = [
    ("00_preflight", "环境与版本分层检查", _preflight),
    ("01_engine_lifecycle", "引擎初始化、预热与稳态延迟", _engine_lifecycle),
    ("02_prefill_decode_sweep", "Prompt/Output 长度二维扫描", _prefill_decode),
    ("03_offline_batching", "离线列表批量与同步逐条调用对照", _offline_batching),
    ("04_sampling_diagnostics", "采样参数、复现性与停止原因", _sampling),
    ("05_kv_pressure", "KV 容量与负载压力递增", _kv_pressure),
    ("06_service_smoke", "OpenAI 兼容服务分层验收", _service_smoke),
    (
        "07_continuous_batching",
        "错峰到达 Continuous Batching 与指标时间线",
        _continuous_batching,
    ),
    ("08_chunked_prefill", "长 Prefill 注入与 Decode ITL 干扰", _chunked_prefill),
    ("09_prefix_caching", "Automatic Prefix Caching 共享前缀对照", _prefix_caching),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def render_document(report_dir: Path) -> str:
    lines = [
        "# vLLM 调试实验结果记录",
        "",
        "> 本文档只记录实验事实，不预填或猜测 RTX 4090 结果。",
        "> 运行实验后执行",
        "> `bash scripts/run_experiment.sh render-results`，表格将由",
        "> `reports/experiments/*.json` 重新生成。该目录默认不提交 Git。",
        "",
        "结果解释必须同时保留模型、vLLM、PyTorch/CUDA、GPU、",
        "引擎参数和负载参数；",
        "缺少这些上下文时，不应跨机器比较绝对延迟或吞吐。",
        "",
    ]
    for experiment_id, title, builder in EXPERIMENTS:
        lines.extend([f"## {experiment_id}：{title}", ""])
        if experiment_id == "08_chunked_prefill":
            paths = sorted(report_dir.glob("08_chunked_prefill_*.json"))
            display_path = "reports/experiments/08_chunked_prefill_*.json"
        else:
            paths = [report_dir / f"{experiment_id}.json"]
            display_path = f"reports/experiments/{experiment_id}.json"
        if not paths or not all(path.exists() for path in paths):
            lines.extend(
                [
                    markdown_table(
                        ["状态", "结果文件", "下一步"],
                        [["未运行", display_path, f"先执行实验 {experiment_id}"]],
                    ),
                    "",
                ]
            )
            continue
        try:
            if experiment_id == "08_chunked_prefill":
                profiles = [
                    json.loads(path.read_text(encoding="utf-8")) for path in paths
                ]
                report = {
                    "status": (
                        "completed"
                        if all(item.get("status") == "completed" for item in profiles)
                        else "failed"
                    ),
                    "config": {
                        item.get("profile", path.stem): item.get(
                            "server_configuration",
                            {},
                        )
                        for path, item in zip(paths, profiles, strict=True)
                    },
                    "profiles": profiles,
                }
            else:
                report = json.loads(paths[0].read_text(encoding="utf-8"))
            headers, rows, notes = builder(report)
            if not rows:
                rows = [["无结果"] + ["N/A"] * (len(headers) - 1)]
            lines.extend(
                [
                    (
                        f"报告状态：`{report.get('status', 'unknown')}`；"
                        f"来源：`{display_path}`。"
                    ),
                    (
                        "配置：`"
                        f"{_detail(report.get('config', report.get('service', {})), 300)}`。"
                    ),
                    "",
                    markdown_table(headers, rows),
                    "",
                ]
            )
            lines.extend(f"- {note}" for note in notes)
            if notes:
                lines.append("")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            lines.extend(
                [
                    markdown_table(
                        ["状态", "结果文件", "错误"],
                        [["无法解析", display_path, f"{type(exc).__name__}: {exc}"]],
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_document(args.report_dir), encoding="utf-8")
    print(f"Rendered: {args.output}")


if __name__ == "__main__":
    main()
