"""Generate numerical CSV evidence for operator validation experiments."""

import argparse
import csv
import math
from pathlib import Path

import torch

import aiop4090 as ops


def metrics(output, reference):
    absolute = (output - reference).abs()
    relative = absolute / reference.abs().clamp_min(1e-7)
    return {
        "max_abs_error": absolute.max().item(),
        "max_rel_error": relative.max().item(),
        "p99_abs_error": torch.quantile(absolute.flatten(), 0.99).item(),
        "nan_count": torch.isnan(output).sum().item(),
        "inf_count": torch.isinf(output).sum().item(),
    }


def layernorm_rows():
    functions = {
        "layernorm_row": ops.layernorm_row,
        "layernorm_block_reduce": ops.layernorm_block_reduce,
        "layernorm_warp_reduce": ops.layernorm_warp_reduce,
        "layernorm_vectorized": ops.layernorm_vectorized,
    }
    cases = [
        ("constant", 1e-3, 257),
        ("constant", 1e-5, 257),
        ("small_variance", 1e-5, 257),
        ("large_offset", 1e-5, 257),
        ("large_offset_aligned", 1e-5, 260),
        ("zero_eps_nonconstant", 0.0, 257),
    ]
    rows = []
    for case, eps, cols in cases:
        gamma = torch.linspace(0.5, 1.5, cols, device="cuda")
        beta = torch.linspace(-0.25, 0.25, cols, device="cuda")
        if case == "constant":
            x = torch.full((3, cols), 7.0, device="cuda")
        elif case == "small_variance":
            x = torch.randn(3, cols, device="cuda") * 1e-5 + 3.0
        elif case.startswith("large_offset"):
            x = torch.randn(3, cols, device="cuda") * 1e-2 + 1e3
        else:
            x = torch.randn(3, cols, device="cuda")
        reference = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps)
        for name, fn in functions.items():
            output = fn(x, gamma, beta, eps)
            result = metrics(output, reference)
            rows.append({
                "experiment": "LN-C02",
                "implementation": name,
                "case": case,
                "shape": f"3x{cols}",
                "eps": eps,
                **result,
                "passed": bool(torch.allclose(output, reference, atol=5e-3, rtol=5e-3)),
            })
    return rows


def rmsnorm_rows():
    functions = {
        "rmsnorm_row": ops.rmsnorm_row,
        "rmsnorm_block_reduce": ops.rmsnorm_block_reduce,
        "rmsnorm_warp_reduce": ops.rmsnorm_warp_reduce,
        "rmsnorm_vectorized": ops.rmsnorm_vectorized,
        "rmsnorm_vectorized_float4": ops.rmsnorm_vectorized_float4,
    }
    rows = []
    cols = 33
    gamma = torch.linspace(0.5, 1.5, cols, device="cuda")
    for eps in (1e-3, 1e-6):
        for value in (0.0, 3.0, 1e10, 1e-10):
            x = torch.full((2, cols), value, device="cuda")
            reference = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma
            for name, fn in functions.items():
                output = fn(x, gamma, eps)
                result = metrics(output, reference)
                rows.append({
                    "experiment": "RMS-C02",
                    "implementation": name,
                    "case": f"constant_{value}",
                    "shape": f"2x{cols}",
                    "eps": eps,
                    **result,
                    "passed": bool(torch.allclose(output, reference, atol=3e-4, rtol=3e-4)),
                })
    return rows


def attention_reference(q, k, v):
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def attention_kv_rows():
    q = torch.randn(1, 2, 1, 32, device="cuda")
    k = torch.randn(1, 2, 256, 32, device="cuda")
    v = torch.randn_like(k)
    rows = []
    for kv_len in (1, 32, 128, 256):
        output = ops.attention_kv_cache_decode(q, k, v, kv_len)
        reference = attention_reference(q, k[:, :, :kv_len], v[:, :, :kv_len])
        result = metrics(output, reference)
        rows.append({
            "experiment": "AT-C03",
            "implementation": "attention_kv_cache_decode",
            "case": f"kv_len_{kv_len}",
            "shape": "1x2x1x32",
            "eps": "N/A",
            **result,
            "passed": bool(torch.allclose(output, reference, atol=4e-4, rtol=4e-4)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["layernorm-numerics", "rmsnorm-numerics", "attention-kv"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=120)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to generate operator validation evidence")
    torch.manual_seed(args.seed)

    generators = {
        "layernorm-numerics": layernorm_rows,
        "rmsnorm-numerics": rmsnorm_rows,
        "attention-kv": attention_kv_rows,
    }
    rows = generators[args.experiment]()
    torch.cuda.synchronize()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['experiment']} {row['implementation']} {row['case']} "
            f"max_abs={row['max_abs_error']:.6g} p99={row['p99_abs_error']:.6g} "
            f"passed={row['passed']}"
        )
    print(f"CSV: {output_path}")
    if not all(row["passed"] for row in rows):
        raise SystemExit("numerical evidence contains a failed case")


if __name__ == "__main__":
    main()
