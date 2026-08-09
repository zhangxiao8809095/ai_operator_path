"""Numerical evidence generator for GEMM-C03 on an RTX 4090."""

import argparse
import csv
from pathlib import Path

import torch

import aiop4090 as ops


def error_metrics(output, reference):
    absolute = (output - reference).abs()
    relative = absolute / reference.abs().clamp_min(1e-7)
    finite = torch.isfinite(output)
    return {
        "max_abs_error": absolute.max().item(),
        "max_rel_error": relative.max().item(),
        "p99_abs_error": torch.quantile(absolute.flatten(), 0.99).item(),
        "nan_count": torch.isnan(output).sum().item(),
        "inf_count": torch.isinf(output).sum().item(),
        "finite_ratio": finite.float().mean().item(),
    }


def run_case(path_name, shape, seed):
    m, n, k = shape
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    output = ops.gemm_wmma_fp16(a, b)
    reference = a.float() @ b.float()
    torch.cuda.synchronize()

    metrics = error_metrics(output, reference)
    atol, rtol = ((5e-2, 2e-2) if path_name == "wmma" else (2e-3, 2e-3))
    passed = bool(torch.allclose(output, reference, atol=atol, rtol=rtol))
    return {
        "path": path_name,
        "m": m,
        "n": n,
        "k": k,
        "input_dtype": str(a.dtype),
        "output_dtype": str(output.dtype),
        **metrics,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Write WMMA error-over-K and fallback evidence to CSV."
    )
    parser.add_argument(
        "--output",
        default="reports/gemm/GEMM-C03.csv",
        help="CSV output path",
    )
    parser.add_argument("--seed", type=int, default=109)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GEMM-C03")

    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise SystemExit(f"WMMA requires compute capability >= 7.0, got {capability}")

    cases = [("wmma", (16, 16, k)) for k in (16, 64, 256, 1024, 4096)]
    cases += [
        ("fallback", (17, 19, 33)),
        ("fallback", (16, 17, 32)),
        ("fallback", (16, 16, 33)),
    ]
    rows = [run_case(path_name, shape, args.seed + index) for index, (path_name, shape) in enumerate(cases)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"GPU: {torch.cuda.get_device_name()}; compute capability: {capability}")
    for row in rows:
        print(
            f"{row['path']:8s} M/N/K={row['m']}/{row['n']}/{row['k']} "
            f"max_abs={row['max_abs_error']:.6g} "
            f"p99_abs={row['p99_abs_error']:.6g} "
            f"max_rel={row['max_rel_error']:.6g} passed={row['passed']}"
        )
    print(f"CSV: {output_path}")

    if not all(row["passed"] for row in rows):
        raise SystemExit("GEMM-C03 failed; inspect the CSV before profiling")


if __name__ == "__main__":
    main()
