#!/usr/bin/env python3
"""Host-only consistency check for the operator-validation handbook and scripts."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = ROOT / "docs/operator_validation_experiments_4090.md"
EXPECTED_EXPORTS = (
    "gemm_naive", "gemm_tiled", "gemm_tiled_padding", "gemm_regtile2x2",
    "gemm_regtile4x4", "gemm_vectorized_float4", "gemm_wmma_fp16",
    "softmax_row", "softmax_block_reduce", "softmax_warp_reduce", "softmax_online",
    "layernorm_row", "layernorm_block_reduce", "layernorm_warp_reduce",
    "layernorm_vectorized", "rmsnorm_row", "rmsnorm_block_reduce",
    "rmsnorm_warp_reduce", "rmsnorm_vectorized", "rmsnorm_vectorized_float4",
    "attention_naive", "attention_causal_naive", "attention_kv_cache_decode",
    "attention_tiled_online_softmax",
)
EXPERIMENTS = {
    "scripts/run_gemm_experiment.sh": (
        "GEMM-C01", "GEMM-C02", "GEMM-C03", "GEMM-D01",
        "GEMM-P01", "GEMM-P02", "GEMM-P03", "GEMM-P04",
    ),
    "scripts/run_operator_experiment.sh": (
        "SM-C01", "SM-C02", "SM-D01", "SM-P01", "SM-P02", "SM-P03",
        "LN-C01", "LN-C02", "LN-C03", "LN-P01", "LN-P02", "LN-P03",
        "RMS-C01", "RMS-C02", "RMS-C03", "RMS-P01", "RMS-P02", "RMS-P03",
        "AT-C01", "AT-C02", "AT-C03", "AT-C04",
        "AT-P01", "AT-P02", "AT-P03", "AT-P04",
    ),
    "scripts/run_debug_experiment.sh": (
        "ENG-C01", "ENG-C02", "ENG-C03", "ENG-D01", "ENG-P01",
        "DBG-E01", "DBG-E02", "DBG-E03", "DBG-E04",
        "DBG-S01", "DBG-S02", "DBG-S03", "DBG-S04",
        "DBG-L01", "DBG-L02", "DBG-L03", "DBG-L04",
        "DBG-T01", "DBG-T02", "DBG-T03",
        "DBG-U01", "DBG-U02", "DBG-U03", "DBG-U04",
    ),
}
GENERATED_PARTS = {".venv", "build", "dist", "reports", "__pycache__"}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_files(suffix: str) -> list[Path]:
    return sorted(
        path for path in ROOT.rglob(f"*{suffix}")
        if not GENERATED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def parser_choices(path: Path, option: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                return {
                    item.value for item in keyword.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
    return set()


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def check_markdown_tables(text: str) -> list[str]:
    failures: list[str] = []
    block: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines() + [""], start=1):
        if line.startswith("|"):
            block.append((line_number, line))
            continue
        if not block:
            continue
        widths = {display_width(row) for _, row in block}
        pipe_counts = {len(re.findall(r"(?<!\\)\|", row)) for _, row in block}
        if len(widths) != 1 or len(pipe_counts) != 1:
            failures.append(
                f"Markdown table {block[0][0]}-{block[-1][0]} is not vertically aligned"
            )
        block = []
    return failures


def run_checks(*, check_generated: bool = True) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    python_files = source_files(".py")
    shell_files = source_files(".sh")

    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            failures.append(f"Python parse failed: {relative(path)}: {error}")
    for path in shell_files:
        completed = subprocess.run(
            ["bash", "-n", str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode:
            failures.append(f"Shell syntax failed: {relative(path)}: {completed.stderr.strip()}")

    junk = sorted(
        relative(path) for path in ROOT.rglob("*")
        if path.name in {".DS_Store", "__pycache__"} or path.suffix in {".pyc", ".pyo"}
    )
    if junk and check_generated:
        failures.append(f"generated local artifacts must not be uploaded: {junk}")

    init_text = (ROOT / "src/aiop4090/__init__.py").read_text(encoding="utf-8")
    binding_text = (ROOT / "src/aiop4090/csrc/bindings.cpp").read_text(encoding="utf-8")
    python_exports = set(re.findall(r"^def ([a-z0-9_]+)\(", init_text, re.MULTILINE))
    binding_exports = set(re.findall(r'm\.def\("([a-z0-9_]+)"', binding_text))
    missing_python = sorted(set(EXPECTED_EXPORTS) - python_exports)
    missing_binding = sorted(set(EXPECTED_EXPORTS) - binding_exports)
    if missing_python or missing_binding:
        failures.append(
            f"24-export contract incomplete: python={missing_python}, binding={missing_binding}"
        )

    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    for source in ("bindings.cpp", "gemm.cu", "softmax.cu", "norm.cu", "attention.cu"):
        if source not in setup_text:
            failures.append(f"setup.py does not compile {source}")

    handbook_text = HANDBOOK.read_text(encoding="utf-8")
    if handbook_text.count("```") % 2:
        failures.append("operator handbook has an unbalanced fenced code block")
    failures.extend(check_markdown_tables(handbook_text))

    for runner, experiment_ids in EXPERIMENTS.items():
        runner_path = ROOT / runner
        if not runner_path.is_file():
            failures.append(f"missing experiment runner: {runner}")
            continue
        runner_text = runner_path.read_text(encoding="utf-8")
        for experiment_id in experiment_ids:
            if experiment_id not in runner_text:
                failures.append(f"{runner} does not implement {experiment_id}")
            command = f"`bash {runner} {experiment_id}`"
            if command not in handbook_text:
                failures.append(f"handbook command missing: {command}")

    command_paths = set(re.findall(
        r"(?:bash|python(?:3)?)\s+((?:scripts|benchmark|debug_labs)/[A-Za-z0-9_./-]+\.(?:sh|py))",
        handbook_text,
    ))
    missing_command_paths = sorted(path for path in command_paths if not (ROOT / path).is_file())
    if missing_command_paths:
        failures.append(f"handbook references missing command files: {missing_command_paths}")

    profile_choices = parser_choices(ROOT / "benchmark/profile_entry.py", "--op")
    documented_profile_ops = set(re.findall(
        r"profile_(?:ncu(?:_full)?|nsys)\.sh\s+([a-z0-9_]+)", handbook_text
    ))
    missing_profile_ops = sorted(documented_profile_ops - profile_choices - {"all"})
    if missing_profile_ops:
        failures.append(f"handbook profile op(s) unsupported by profile_entry.py: {missing_profile_ops}")

    required_executables = {
        "scripts/00_check_env.sh", "scripts/10_build.sh", "scripts/15_verify_4090.sh",
        "scripts/20_test.sh", "scripts/30_bench.sh", "scripts/40_sanitize.sh",
        "scripts/50_debug_labs.sh", *EXPERIMENTS.keys(),
    }
    non_executable = sorted(
        path for path in required_executables
        if (ROOT / path).is_file() and not os.access(ROOT / path, os.X_OK)
    )
    if non_executable:
        failures.append(f"entrypoint is not executable: {non_executable}")

    result = {
        "root": str(ROOT),
        "python_file_count": len(python_files),
        "shell_file_count": len(shell_files),
        "formal_export_count": len(EXPECTED_EXPORTS),
        "documented_command_file_count": len(command_paths),
        "documented_profile_op_count": len(documented_profile_ops),
        "experiment_count": sum(len(items) for items in EXPERIMENTS.values()),
        "generated_artifact_count": len(junk),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    return result, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-generated", action="store_true")
    args = parser.parse_args()
    result, failures = run_checks(check_generated=not args.allow_generated)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.json:
        destination = args.json if args.json.is_absolute() else ROOT / args.json
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
