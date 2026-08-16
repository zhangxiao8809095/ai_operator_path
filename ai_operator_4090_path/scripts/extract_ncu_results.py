#!/usr/bin/env python3
"""Batch-extract NCU reports into tables that can be compared without NCU UI.

The existing shell extractors remain the single-report backend.  This tool adds
report-name to kernel-name mapping, family filtering, derived/source-known
metrics, and wide CSV/Markdown summaries for the validation handbook.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIXED_EXTRACTOR = PROJECT_DIR / "scripts" / "extract_ncu_metrics.sh"
SUPPLEMENTAL_EXTRACTOR = PROJECT_DIR / "scripts" / "extract_ncu_supplemental_metrics.sh"

FAMILIES = ("gemm", "softmax", "layernorm", "rmsnorm", "attention")
FIXED_COLUMNS = (
    "Duration",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "DRAM Throughput",
    "L2 Cache Throughput",
    "Achieved Occupancy",
    "Registers / Thread",
    "Top Stall Reason",
)
FIXED_SOURCE_NAMES = {
    "Duration": "Duration",
    "Compute (SM) Throughput": "Compute Throughput",
    "Memory Throughput": "Memory Throughput",
    "DRAM Throughput": "DRAM Throughput",
    "L2 Cache Throughput": "L2 Throughput",
    "Achieved Occupancy": "Achieved Occupancy",
    "Registers / Thread": "Registers / Thread",
    "Top Stall Reason": "Top Stall Reason",
}


@dataclass(frozen=True)
class ProfileSpec:
    family: str
    version: str
    kernels: tuple[str, ...]
    shape: str
    gemm_shape: tuple[int, int, int] | None = None
    source_metrics: Mapping[str, str] = field(default_factory=dict)


def spec(
    family: str,
    version: str,
    kernel: str | Sequence[str],
    shape: str,
    *,
    gemm_shape: tuple[int, int, int] | None = None,
    source_metrics: Mapping[str, str] | None = None,
) -> ProfileSpec:
    kernels = (kernel,) if isinstance(kernel, str) else tuple(kernel)
    return ProfileSpec(
        family=family,
        version=version,
        kernels=kernels,
        shape=shape,
        gemm_shape=gemm_shape,
        source_metrics=dict(source_metrics or {}),
    )


ATTENTION_TRUE_KERNELS = (
    "attention_naive_kernel",
)
ATTENTION_FALSE_KERNELS = (
    "attention_naive_kernel",
)

REGISTRY: dict[str, ProfileSpec] = {
    # GEMM formal versions and path-specific scenarios.
    "gemm_naive": spec("gemm", "gemm_naive", "gemm_naive_kernel", "M=N=K=2048", gemm_shape=(2048, 2048, 2048)),
    "gemm_tiled": spec("gemm", "gemm_tiled", "gemm_tiled_kernel", "M=N=K=2048", gemm_shape=(2048, 2048, 2048)),
    "gemm_tiled_padding": spec("gemm", "gemm_tiled_padding", "gemm_tiled_padding_kernel", "M=N=K=2048", gemm_shape=(2048, 2048, 2048)),
    "gemm_regtile2x2": spec("gemm", "gemm_regtile2x2", "gemm_regtile2x2_kernel", "M=N=K=2048", gemm_shape=(2048, 2048, 2048)),
    "gemm_regtile4x4": spec("gemm", "gemm_regtile4x4", "gemm_regtile4x4_kernel", "M=N=K=2048", gemm_shape=(2048, 2048, 2048)),
    "gemm_vectorized_float4": spec("gemm", "gemm_vectorized_float4", "gemm_vectorized_float4_kernel", "M=N=K=2048, aligned", gemm_shape=(2048, 2048, 2048)),
    "gemm_vectorized_float4_misaligned": spec("gemm", "gemm_vectorized_float4", "gemm_vectorized_float4_kernel", "M=N=K=2048, B misaligned", gemm_shape=(2048, 2048, 2048)),
    "gemm_vectorized_float4_tail": spec("gemm", "gemm_vectorized_float4", "gemm_vectorized_float4_kernel", "M=2048,N=2047,K=2048", gemm_shape=(2048, 2047, 2048)),
    "gemm_wmma_fp16": spec("gemm", "gemm_wmma_fp16", "gemm_wmma_fp16_kernel", "M=N=K=2048, FP16", gemm_shape=(2048, 2048, 2048)),
    "gemm_wmma_from_fp32": spec("gemm", "gemm_wmma_fp16", "gemm_wmma_fp16_kernel", "M=N=K=2048, FP32 input conversion", gemm_shape=(2048, 2048, 2048)),
    "gemm_wmma_fallback": spec("gemm", "gemm_wmma_fp16", "gemm_tiled_kernel", "M=2033,N=2047,K=2049, fallback", gemm_shape=(2033, 2047, 2049)),
    # Softmax. Input-pass counts are source-derived, not NCU counters.
    "softmax": spec("softmax", "softmax_row", "softmax_row_kernel", "rows=8192, cols=4096", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_row": spec("softmax", "softmax_row", "softmax_row_kernel", "rows=8192, cols=4096", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_block_reduce": spec("softmax", "softmax_block_reduce", "softmax_block_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_warp_reduce": spec("softmax", "softmax_warp_reduce", "softmax_warp_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_online": spec("softmax", "softmax_online", "softmax_online_kernel", "rows=8192, cols=4096", source_metrics={"Input Passes": "2 (source-derived)"}),
    "softmax_row_small": spec("softmax", "softmax_row", "softmax_row_kernel", "rows=4, cols=33", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_block_reduce_small": spec("softmax", "softmax_block_reduce", "softmax_block_reduce_kernel", "rows=4, cols=33", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_warp_reduce_small": spec("softmax", "softmax_warp_reduce", "softmax_warp_reduce_kernel", "rows=4, cols=33", source_metrics={"Input Passes": "3 (source-derived)"}),
    "softmax_online_small": spec("softmax", "softmax_online", "softmax_online_kernel", "rows=4, cols=33", source_metrics={"Input Passes": "2 (source-derived)"}),
    # LayerNorm aligned and fallback paths.
    "layernorm": spec("layernorm", "layernorm_row", "layernorm_row_kernel", "rows=8192, cols=4096", source_metrics={"Float4 Load/Store Path": "scalar row kernel"}),
    "layernorm_row": spec("layernorm", "layernorm_row", "layernorm_row_kernel", "rows=8192, cols=4096", source_metrics={"Float4 Load/Store Path": "scalar row kernel"}),
    "layernorm_block_reduce": spec("layernorm", "layernorm_block_reduce", "layernorm_block_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Float4 Load/Store Path": "scalar block kernel"}),
    "layernorm_warp_reduce": spec("layernorm", "layernorm_warp_reduce", "layernorm_warp_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Float4 Load/Store Path": "scalar warp kernel"}),
    "layernorm_vectorized": spec("layernorm", "layernorm_vectorized", "layernorm_vectorized_kernel", "rows=8192, cols=4096, aligned", source_metrics={"Float4 Load/Store Path": "float4 kernel (source-derived)"}),
    "layernorm_vectorized_misaligned": spec("layernorm", "layernorm_vectorized", "layernorm_warp_reduce_kernel", "rows=8192, cols=4096, misaligned fallback", source_metrics={"Float4 Load/Store Path": "warp fallback (source-derived)"}),
    "layernorm_vectorized_tail": spec("layernorm", "layernorm_vectorized", "layernorm_warp_reduce_kernel", "rows=8192, cols=4098, tail fallback", source_metrics={"Float4 Load/Store Path": "warp fallback (source-derived)"}),
    # RMSNorm aligned and fallback paths.
    "rmsnorm": spec("rmsnorm", "rmsnorm_row", "rmsnorm_row_kernel", "rows=8192, cols=4096", source_metrics={"Float2/Float4 Width": "scalar row kernel"}),
    "rmsnorm_row": spec("rmsnorm", "rmsnorm_row", "rmsnorm_row_kernel", "rows=8192, cols=4096", source_metrics={"Float2/Float4 Width": "scalar row kernel"}),
    "rmsnorm_block_reduce": spec("rmsnorm", "rmsnorm_block_reduce", "rmsnorm_block_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Float2/Float4 Width": "scalar block kernel"}),
    "rmsnorm_warp_reduce": spec("rmsnorm", "rmsnorm_warp_reduce", "rmsnorm_warp_reduce_kernel", "rows=8192, cols=4096", source_metrics={"Float2/Float4 Width": "scalar warp kernel"}),
    "rmsnorm_vectorized": spec("rmsnorm", "rmsnorm_vectorized", "rmsnorm_vectorized_float2_kernel", "rows=8192, cols=4096, aligned", source_metrics={"Float2/Float4 Width": "float2 kernel (source-derived)"}),
    "rmsnorm_vectorized_float4": spec("rmsnorm", "rmsnorm_vectorized_float4", "rmsnorm_vectorized_kernel", "rows=8192, cols=4096, aligned", source_metrics={"Float2/Float4 Width": "float4 kernel (source-derived)"}),
    "rmsnorm_vectorized_float2_only": spec("rmsnorm", "rmsnorm_vectorized", "rmsnorm_vectorized_float2_kernel", "rows=8192, cols=4098", source_metrics={"Float2/Float4 Width": "float2-only shape (source-derived)"}),
    "rmsnorm_vectorized_float4_fallback": spec("rmsnorm", "rmsnorm_vectorized_float4", "rmsnorm_warp_reduce_kernel", "rows=8192, cols=4098, fallback", source_metrics={"Float2/Float4 Width": "warp fallback (source-derived)"}),
    "rmsnorm_vectorized_float4_misaligned": spec("rmsnorm", "rmsnorm_vectorized_float4", "rmsnorm_warp_reduce_kernel", "rows=8192, cols=4096, misaligned fallback", source_metrics={"Float2/Float4 Width": "warp fallback (source-derived)"}),
    # Attention prefill/decode/causal scenarios.
    "attention_naive": spec("attention", "attention_naive", ATTENTION_TRUE_KERNELS, "B=1,H=8,S=128,D=64, causal", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_naive_noncausal": spec("attention", "attention_naive", ATTENTION_FALSE_KERNELS, "B=1,H=8,S=128,D=64, non-causal", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_causal_naive": spec("attention", "attention_causal_naive", ATTENTION_TRUE_KERNELS, "B=1,H=8,S=128,D=64, compile-time causal", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_naive_s64": spec("attention", "attention_naive", ATTENTION_TRUE_KERNELS, "B=1,H=8,S=64,D=64, causal", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_tiled_online_s64": spec("attention", "attention_tiled_online_softmax", "attention_tiled_online_softmax_kernel", "B=1,H=8,S=64,D=64, causal", source_metrics={"Score Intermediate": "not materialized; online state (source-derived)"}),
    "attention_kv_cache_decode": spec("attention", "attention_kv_cache_decode", "attention_kv_cache_decode_kernel", "B=1,H=8,Q=1,D=64, kv_len=128", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_kv_cache_decode_kv1": spec("attention", "attention_kv_cache_decode", "attention_kv_cache_decode_kernel", "B=1,H=8,Q=1,D=64, kv_len=1", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_kv_cache_decode_kv32": spec("attention", "attention_kv_cache_decode", "attention_kv_cache_decode_kernel", "B=1,H=8,Q=1,D=64, kv_len=32", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_kv_cache_decode_kv128": spec("attention", "attention_kv_cache_decode", "attention_kv_cache_decode_kernel", "B=1,H=8,Q=1,D=64, kv_len=128", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_kv_cache_decode_kv256": spec("attention", "attention_kv_cache_decode", "attention_kv_cache_decode_kernel", "B=1,H=8,Q=1,D=64, kv_len=256", source_metrics={"Score Intermediate": "not materialized; score recomputed (source-derived)"}),
    "attention_tiled_online_softmax": spec("attention", "attention_tiled_online_softmax", "attention_tiled_online_softmax_kernel", "B=1,H=8,S=128,D=64, causal", source_metrics={"Score Intermediate": "not materialized; online state (source-derived)"}),
    "attention_tiled_online_softmax_noncausal": spec("attention", "attention_tiled_online_softmax", "attention_tiled_online_softmax_kernel", "B=1,H=8,S=128,D=64, non-causal", source_metrics={"Score Intermediate": "not materialized; online state (source-derived)"}),
}


FOCUS_METRICS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "gemm": (
        ("Achieved TFLOP/s", ("Achieved TFLOP/s",)),
        ("L2 Absolute Traffic", ("L2 / DRAM Bytes",)),
        ("Shared Bank Conflicts", ("Shared Bank Conflicts",)),
        ("Register Spill", ("Local Load/Store",)),
        ("FP32/Tensor Pipeline", ("FMA Pipe Utilization", "Tensor Pipe Utilization")),
        ("MMA Instructions", ("Tensor / HMMA Instructions",)),
    ),
    "softmax": (
        ("Input Passes", ("Input Passes",)),
        ("Load Bytes", ("L2 / DRAM Bytes",)),
        ("Exp/Arithmetic", ("MUFU / SFU Instructions", "FP Arithmetic Instructions")),
        ("Shared Access", ("Shared Load Instructions", "Shared Store Instructions", "Shared Bank Conflicts")),
        ("Barrier Wait", ("Barrier Stall",)),
        ("MIO/Short Scoreboard", ("MIO Throttle Stall", "Short Scoreboard Stall")),
        ("Launch/End-to-end", ("Launch/End-to-end",)),
    ),
    "layernorm": (
        ("Mean/Variance Reduction", ("Mean/Variance Reduction",)),
        ("Shared/Barrier", ("Shared Load Instructions", "Shared Store Instructions", "Barrier Stall")),
        ("Float4 Load/Store Path", ("Float4 Load/Store Path",)),
        ("Requests/Sectors", ("Global Requests", "Global Sectors")),
        ("Absolute Bytes", ("L2 / DRAM Bytes",)),
    ),
    "rmsnorm": (
        ("RMS Reduction Work", ("RMS Reduction Work",)),
        ("Shared/Barrier", ("Shared Load Instructions", "Shared Store Instructions", "Barrier Stall")),
        ("Float2/Float4 Width", ("Float2/Float4 Width",)),
        ("Requests/Sectors", ("Global Requests", "Global Sectors")),
        ("Absolute Read/Write Bytes", ("L2 / DRAM Bytes",)),
    ),
    "attention": (
        ("Q/K/V and Cache Bytes", ("L2 / DRAM Bytes",)),
        ("L2 Hit Rate/Absolute Traffic", ("L2 Hit Rate", "L2 / DRAM Bytes")),
        ("Score Intermediate", ("Score Intermediate",)),
        ("Register Spill", ("Local Load/Store",)),
        ("Long Scoreboard", ("Long Scoreboard Stall",)),
        ("Causal Branch Efficiency", ("Branch Instructions", "Uniform Branch Targets")),
        ("Duration vs S/kv_len", ("Duration",)),
    ),
}

UNAVAILABLE_NOTES = {
    "Launch/End-to-end": "N/A (requires NSYS, not a kernel-only NCU report)",
    "Mean/Variance Reduction": "N/A (source/SASS-derived; compare instruction work)",
    "RMS Reduction Work": "N/A (source/SASS-derived; compare instruction work)",
}


@dataclass
class Result:
    report: Path
    profile_op: str
    spec: ProfileSpec
    kernel: str
    invocation: int
    fixed: dict[str, str]
    supplemental: dict[str, str]


def parse_markdown_metrics(text: str) -> dict[str, str]:
    """Parse the two-column Markdown table emitted by the shell extractors."""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key in {"Metric", "Supplemental Metric"}:
            continue
        if key and set(key) <= {"-", ":"}:
            continue
        parsed[key] = value
    return parsed


def duration_seconds(value: str) -> float | None:
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([a-zA-Z\u00b5]+)", value)
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower().replace("\u00b5", "u")
    scales = {
        "ns": 1e-9,
        "nsecond": 1e-9,
        "us": 1e-6,
        "usecond": 1e-6,
        "ms": 1e-3,
        "msecond": 1e-3,
        "s": 1.0,
        "second": 1.0,
    }
    scale = scales.get(unit)
    return None if scale is None else number * scale


def report_profile_op(path: Path) -> str:
    name = path.name
    if not name.endswith(".ncu-rep"):
        raise ValueError(f"not an NCU report: {path}")
    stem = name[: -len(".ncu-rep")]
    return stem[: -len("_full")] if stem.endswith("_full") else stem


def find_ncu(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env_bin = os.environ.get("NCU_BIN")
    if env_bin:
        candidates.append(env_bin)
    path_bin = shutil.which("ncu")
    if path_bin:
        candidates.append(path_bin)
    candidates.extend(
        str(path)
        for path in (
            Path("/usr/local/cuda/bin/ncu"),
            Path("/usr/local/cuda-12.6/bin/ncu"),
            Path("/usr/local/cuda-12.4/bin/ncu"),
            Path("/usr/local/cuda-12.1/bin/ncu"),
        )
    )
    for candidate in candidates:
        executable = shutil.which(candidate)
        path = Path(executable or candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError("ncu was not found; add it to PATH, set NCU_BIN, or pass --ncu-bin")


def run_extractor(script: Path, report: Path, kernel: str, invocation: int, ncu_bin: str) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["NCU_BIN"] = ncu_bin
    completed = subprocess.run(
        ["bash", str(script), str(report), kernel, str(invocation)],
        cwd=PROJECT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def extract_report(report: Path, profile_op: str, profile_spec: ProfileSpec, invocation: int, ncu_bin: str) -> Result:
    failures: list[str] = []
    invocation_candidates = (invocation,) if invocation == 1 else (invocation, 1)
    for kernel in profile_spec.kernels:
        for selected_invocation in invocation_candidates:
            status, fixed_stdout, fixed_stderr = run_extractor(
                FIXED_EXTRACTOR, report, kernel, selected_invocation, ncu_bin
            )
            if status != 0:
                failures.append(
                    f"{kernel} invocation={selected_invocation}: "
                    f"{fixed_stderr.strip() or 'fixed extractor failed'}"
                )
                continue

            status, supplemental_stdout, supplemental_stderr = run_extractor(
                SUPPLEMENTAL_EXTRACTOR, report, kernel, selected_invocation, ncu_bin
            )
            if status != 0:
                failures.append(
                    f"{kernel} invocation={selected_invocation}: "
                    f"{supplemental_stderr.strip() or 'supplemental extractor failed'}"
                )
                continue

            raw_fixed = parse_markdown_metrics(fixed_stdout)
            raw_supplemental = parse_markdown_metrics(supplemental_stdout)
            fixed = {
                output_name: raw_fixed.get(source_name, "N/A (metric unavailable)")
                for output_name, source_name in FIXED_SOURCE_NAMES.items()
            }
            supplemental = dict(raw_supplemental)
            supplemental.update(profile_spec.source_metrics)
            if profile_spec.family == "layernorm":
                supplemental.setdefault(
                    "Mean/Variance Reduction",
                    "1 shifted-Welford state reduction then normalize (source-derived)",
                )
            if profile_spec.family == "rmsnorm":
                supplemental.setdefault(
                    "RMS Reduction Work",
                    "1 sum-of-squares reduction then normalize (source-derived)",
                )
            for name, note in UNAVAILABLE_NOTES.items():
                supplemental.setdefault(name, note)

            if profile_spec.gemm_shape:
                seconds = duration_seconds(fixed["Duration"])
                if seconds and seconds > 0:
                    m, n, k = profile_spec.gemm_shape
                    supplemental["Achieved TFLOP/s"] = f"{2.0 * m * n * k / seconds / 1e12:.3f} TFLOP/s"
                else:
                    supplemental["Achieved TFLOP/s"] = "N/A (Duration unit could not be parsed)"

            # Duration is also a family-specific series for Attention.
            supplemental["Duration"] = fixed["Duration"]
            return Result(
                report=report,
                profile_op=profile_op,
                spec=profile_spec,
                kernel=kernel,
                invocation=selected_invocation,
                fixed=fixed,
                supplemental=supplemental,
            )

    details = "\n  ".join(failures)
    raise RuntimeError(f"no kernel candidate matched {report}:\n  {details}")


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def display_width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value)


def pad_display(value: str, width: int) -> str:
    return value + " " * max(0, width - display_width(value))


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    escaped_headers = [markdown_escape(item) for item in headers]
    escaped_rows = [[markdown_escape(item) for item in row] for row in rows]
    widths = [display_width(header) for header in escaped_headers]
    for row in escaped_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], display_width(cell))
    lines = [
        "| " + " | ".join(pad_display(cell, widths[index]) for index, cell in enumerate(escaped_headers)) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(widths))) + " |",
    ]
    for row in escaped_rows:
        lines.append(
            "| " + " | ".join(pad_display(cell, widths[index]) for index, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def focus_value(result: Result, sources: Sequence[str]) -> str:
    values: list[str] = []
    for source in sources:
        if source in result.supplemental:
            value = result.supplemental[source]
        elif source in result.fixed:
            value = result.fixed[source]
        else:
            value = "N/A (metric unavailable)"
        values.append(value if len(sources) == 1 else f"{source}={value}")
    return "; ".join(values)


def write_wide_csv(path: Path, results: Sequence[Result], columns: Sequence[str], source: str) -> None:
    metadata = ["family", "profile_op", "operator_version", "shape_or_scenario", "kernel", "invocation", "report"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata + list(columns))
        writer.writeheader()
        for result in results:
            row = {
                "family": result.spec.family,
                "profile_op": result.profile_op,
                "operator_version": result.spec.version,
                "shape_or_scenario": result.spec.shape,
                "kernel": result.kernel,
                "invocation": result.invocation,
                "report": str(result.report),
            }
            values = result.fixed if source == "fixed" else result.supplemental
            row.update({column: values.get(column, "N/A (metric unavailable)") for column in columns})
            writer.writerow(row)


def render_summary(family: str, results: Sequence[Result]) -> str:
    fixed_headers = ("Profile scenario", "Operator version", "Shape/scenario", *FIXED_COLUMNS)
    fixed_rows = [
        (
            result.profile_op,
            result.spec.version,
            result.spec.shape,
            *(result.fixed[column] for column in FIXED_COLUMNS),
        )
        for result in results
    ]
    focus = FOCUS_METRICS[family]
    focus_headers = ("Profile scenario", "Operator version", "Shape/scenario", *(name for name, _ in focus))
    focus_rows = [
        (
            result.profile_op,
            result.spec.version,
            result.spec.shape,
            *(focus_value(result, sources) for _, sources in focus),
        )
        for result in results
    ]
    return (
        f"# {family} NCU extraction summary\n\n"
        "`N/A` 表示报告中没有该指标或该证据需要 NSYS/源码/SASS，不能按 0 解释。\n\n"
        "## Fixed eight metrics\n\n"
        f"{markdown_table(fixed_headers, fixed_rows)}\n\n"
        "## Operator-specific metrics\n\n"
        f"{markdown_table(focus_headers, focus_rows)}\n"
    )


def supplemental_columns(results: Sequence[Result]) -> list[str]:
    priority: list[str] = []
    for family in FAMILIES:
        for _, sources in FOCUS_METRICS[family]:
            for source in sources:
                if source not in FIXED_COLUMNS and source not in priority:
                    priority.append(source)
    discovered = sorted({name for result in results for name in result.supplemental})
    return priority + [name for name in discovered if name not in priority]


def write_outputs(output_dir: Path, results: Sequence[Result]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    grouped = {family: [result for result in results if result.spec.family == family] for family in FAMILIES}
    for family, family_results in grouped.items():
        if not family_results:
            continue
        fixed_path = output_dir / f"{family}_fixed8.csv"
        supplemental_path = output_dir / f"{family}_supplemental.csv"
        summary_path = output_dir / f"{family}_summary.md"
        write_wide_csv(fixed_path, family_results, FIXED_COLUMNS, "fixed")
        write_wide_csv(
            supplemental_path,
            family_results,
            supplemental_columns(family_results),
            "supplemental",
        )
        summary_path.write_text(render_summary(family, family_results), encoding="utf-8")
        written.extend((fixed_path, supplemental_path, summary_path))
    return written


def discover_reports(report: Path | None, report_dir: Path | None) -> list[Path]:
    if report:
        if not report.is_file():
            raise FileNotFoundError(f"report not found: {report}")
        return [report]
    if report_dir is None or not report_dir.is_dir():
        raise FileNotFoundError(f"report directory not found: {report_dir}")
    return sorted(report_dir.glob("*_full.ncu-rep"))


def print_mappings() -> None:
    rows = [
        (name, item.family, item.version, item.shape, " OR ".join(item.kernels))
        for name, item in sorted(REGISTRY.items())
    ]
    print(markdown_table(("Profile op", "Family", "Operator version", "Shape/scenario", "Kernel candidate"), rows))


def self_test() -> None:
    sample = """
| Metric                       | Value                                                |
| ---------------------------- | ---------------------------------------------------- |
| Duration                     | 125.00 usecond                                       |
| Compute Throughput           | 50.00 %                                              |
| Top Stall Reason             | Long Scoreboard (2.00 cycles, 20.00%)                |
"""
    parsed = parse_markdown_metrics(sample)
    assert parsed["Duration"] == "125.00 usecond"
    assert parsed["Top Stall Reason"].startswith("Long Scoreboard")
    assert abs((duration_seconds("125.00 usecond") or 0.0) - 125e-6) < 1e-12
    assert report_profile_op(Path("softmax_online_full.ncu-rep")) == "softmax_online"
    assert REGISTRY["attention_naive"].family == "attention"
    table = markdown_table(("算子", "Duration"), (("softmax", "1 us"),))
    assert "| 算子" in table and "softmax" in table
    demo = Result(
        report=Path("reports/ncu/softmax_online_full.ncu-rep"),
        profile_op="softmax_online",
        spec=REGISTRY["softmax_online"],
        kernel="softmax_online_kernel",
        invocation=6,
        fixed={column: "1" for column in FIXED_COLUMNS},
        supplemental={"Input Passes": "2 (source-derived)", "Duration": "1 usecond"},
    )
    with tempfile.TemporaryDirectory() as directory:
        outputs = write_outputs(Path(directory), [demo])
        assert {path.name for path in outputs} == {
            "softmax_fixed8.csv",
            "softmax_supplemental.csv",
            "softmax_summary.md",
        }
        assert "Input Passes" in (Path(directory) / "softmax_summary.md").read_text(encoding="utf-8")
    print("extract_ncu_results.py self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-extract NCU Full reports into wide CSV and Markdown tables."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--report", type=Path, help="extract one .ncu-rep file")
    source.add_argument("--report-dir", type=Path, help="scan *_full.ncu-rep files in a directory")
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ncu_summary"))
    parser.add_argument(
        "--invocation", type=int, default=6,
        help="preferred kernel invocation after five warmups; batch extraction falls back to 1",
    )
    parser.add_argument("--ncu-bin", help="path to ncu; otherwise use NCU_BIN/PATH/CUDA defaults")
    parser.add_argument("--strict", action="store_true", help="fail when the directory contains an unknown report name")
    parser.add_argument("--dry-run", action="store_true", help="show report-to-kernel mapping without invoking ncu")
    parser.add_argument("--list-mappings", action="store_true", help="print all supported profile report mappings")
    parser.add_argument("--self-test", action="store_true", help="run host-only parser and registry checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.list_mappings:
        print_mappings()
        return 0
    if args.invocation < 1:
        raise SystemExit("error: --invocation must be a positive integer")
    if args.report is None and args.report_dir is None:
        args.report_dir = Path("reports/ncu")

    reports = discover_reports(args.report, args.report_dir)
    plans: list[tuple[Path, str, ProfileSpec]] = []
    unknown: list[Path] = []
    for report in reports:
        profile_op = report_profile_op(report)
        profile_spec = REGISTRY.get(profile_op)
        if profile_spec is None:
            unknown.append(report)
            continue
        if args.family != "all" and profile_spec.family != args.family:
            continue
        plans.append((report, profile_op, profile_spec))

    if unknown:
        message = "unknown report name(s): " + ", ".join(str(path) for path in unknown)
        if args.strict:
            raise SystemExit(f"error: {message}; use --list-mappings")
        print(f"warning: {message}; skipped", file=sys.stderr)
    if not plans:
        raise SystemExit("error: no matching Full reports were found")

    if args.dry_run:
        rows = [
            (path, profile_op, item.family, item.version, item.shape, " OR ".join(item.kernels))
            for path, profile_op, item in plans
        ]
        print(markdown_table(("Report", "Profile op", "Family", "Version", "Shape", "Kernel candidate"), rows))
        return 0

    ncu_bin = find_ncu(args.ncu_bin)
    results: list[Result] = []
    for index, (report, profile_op, profile_spec) in enumerate(plans, start=1):
        print(f"[{index}/{len(plans)}] extracting {profile_op} ...", file=sys.stderr)
        results.append(extract_report(report, profile_op, profile_spec, args.invocation, ncu_bin))

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    written = write_outputs(output_dir, results)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
