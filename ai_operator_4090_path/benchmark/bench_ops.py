import argparse
import csv
import math
from pathlib import Path
import statistics
import torch
import aiop4090 as ops


def cuda_bench(fn, warmup=20, repeat=100, rounds=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeat)
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def format_stats(stats):
    spread = (stats["max"] - stats["min"]) / stats["median"] * 100.0
    return (
        f"median={stats['median']:8.4f} ms "
        f"p90={stats['p90']:8.4f} min={stats['min']:8.4f} max={stats['max']:8.4f} "
        f"spread={spread:6.2f}%"
    )


def record_result(records, family, shape, name, stats, throughput_tflops=None):
    records.append({
        "family": family,
        "shape": shape,
        "implementation": name,
        "median_ms": stats["median"],
        "p90_ms": stats["p90"],
        "min_ms": stats["min"],
        "max_ms": stats["max"],
        "spread_pct": (stats["max"] - stats["min"]) / stats["median"] * 100.0,
        "throughput_tflops": "" if throughput_tflops is None else throughput_tflops,
    })


def bench_gemm(records):
    print("\n[GEMM] ms/op")
    for m, n, k in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        a_half = a.half()
        b_half = b.half()
        cases = [
            ("torch_mm", lambda: a @ b),
            ("naive", lambda: ops.gemm_naive(a, b)),
            ("tiled", lambda: ops.gemm_tiled(a, b)),
            ("tiled_padding", lambda: ops.gemm_tiled_padding(a, b)),
            ("regtile2x2", lambda: ops.gemm_regtile2x2(a, b)),
            ("regtile4x4", lambda: ops.gemm_regtile4x4(a, b)),
            ("vector_float4", lambda: ops.gemm_vectorized_float4(a, b)),
            ("wmma_fp16", lambda: ops.gemm_wmma_fp16(a_half, b_half)),
        ]
        for name, fn in cases:
            stats = cuda_bench(fn, warmup=10, repeat=30)
            tflops = (2 * m * n * k) / (stats["median"] * 1e-3) / 1e12
            print(f"shape=({m},{n},{k}) {name:12s}: {format_stats(stats)}, {tflops:7.3f} TFLOP/s")
            record_result(records, "gemm", f"{m}x{n}x{k}", name, stats, tflops)


def bench_softmax_norm(records, families=None):
    print("\n[Softmax/Norm] ms/op")
    for rows, cols in [(4096, 1024), (4096, 4096), (8192, 4096)]:
        x = torch.randn(rows, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        beta = torch.randn(cols, device="cuda")
        cases = [
            ("softmax", "torch_softmax", lambda: torch.softmax(x, dim=-1)),
            ("softmax", "softmax_row", lambda: ops.softmax_row(x)),
            ("softmax", "softmax_block_reduce", lambda: ops.softmax_block_reduce(x)),
            ("softmax", "softmax_warp_reduce", lambda: ops.softmax_warp_reduce(x)),
            ("softmax", "softmax_online", lambda: ops.softmax_online(x)),
            ("layernorm", "torch_layernorm", lambda: torch.nn.functional.layer_norm(x, (cols,), gamma, beta)),
            ("layernorm", "layernorm_row", lambda: ops.layernorm_row(x, gamma, beta)),
            ("layernorm", "layernorm_block_reduce", lambda: ops.layernorm_block_reduce(x, gamma, beta)),
            ("layernorm", "layernorm_warp_reduce", lambda: ops.layernorm_warp_reduce(x, gamma, beta)),
            ("layernorm", "layernorm_vectorized", lambda: ops.layernorm_vectorized(x, gamma, beta)),
            ("rmsnorm", "rmsnorm_row", lambda: ops.rmsnorm_row(x, gamma)),
            ("rmsnorm", "rmsnorm_block_reduce", lambda: ops.rmsnorm_block_reduce(x, gamma)),
            ("rmsnorm", "rmsnorm_warp_reduce", lambda: ops.rmsnorm_warp_reduce(x, gamma)),
            ("rmsnorm", "rmsnorm_vectorized", lambda: ops.rmsnorm_vectorized(x, gamma)),
            ("rmsnorm", "rmsnorm_vectorized_float4", lambda: ops.rmsnorm_vectorized_float4(x, gamma)),
        ]
        for family, name, fn in cases:
            if families is not None and family not in families:
                continue
            stats = cuda_bench(fn, warmup=10, repeat=50)
            print(f"shape=({rows},{cols}) {name:26s}: {format_stats(stats)}")
            record_result(records, family, f"{rows}x{cols}", name, stats)


def bench_attention(records):
    print("\n[Attention naive] ms/op")
    # Keep shapes modest. The included naive kernel intentionally recomputes dot-products.
    for b, h, s, d in [(1, 4, 64, 64), (1, 8, 128, 64)]:
        q = torch.randn(b, h, s, d, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cases = [
            ("torch_sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)),
            ("attention_naive", lambda: ops.attention_naive(q, k, v, True)),
            ("attention_causal_naive", lambda: ops.attention_causal_naive(q, k, v)),
            ("attention_tiled_online", lambda: ops.attention_tiled_online_softmax(q, k, v, True)),
        ]
        for name, fn in cases:
            stats = cuda_bench(fn, warmup=5, repeat=20)
            print(f"shape=({b},{h},{s},{d}) {name:24s}: {format_stats(stats)}")
            record_result(records, "attention", f"{b}x{h}x{s}x{d}", name, stats)

    print("\n[Attention KV-cache decode] ms/op")
    for b, h, cache_s, d in [(1, 8, 128, 64), (2, 8, 256, 64)]:
        q = torch.randn(b, h, 1, d, device="cuda")
        k_cache = torch.randn(b, h, cache_s, d, device="cuda")
        v_cache = torch.randn_like(k_cache)
        for kv_len in sorted({1, cache_s // 2, cache_s}):
            name = "kv_cache_decode"
            fn = lambda kv_len=kv_len: ops.attention_kv_cache_decode(q, k_cache, v_cache, kv_len)
            stats = cuda_bench(fn, warmup=5, repeat=20)
            print(f"shape=({b},{h},1,{d}) kv_len={kv_len} {name:16s}: {format_stats(stats)}")
            record_result(records, "attention_decode", f"{b}x{h}x1x{d};kv={kv_len}", name, stats)


def write_csv(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        choices=["all", "gemm", "softmax_norm", "softmax", "layernorm", "rmsnorm", "norm", "attention"],
    )
    parser.add_argument("--csv", help="optional output CSV path")
    args = parser.parse_args()
    torch.manual_seed(0)
    records = []
    if args.op in ("all", "gemm"):
        bench_gemm(records)
    if args.op in ("all", "softmax_norm"):
        bench_softmax_norm(records)
    elif args.op == "norm":
        bench_softmax_norm(records, {"layernorm", "rmsnorm"})
    elif args.op in ("softmax", "layernorm", "rmsnorm"):
        bench_softmax_norm(records, {args.op})
    if args.op in ("all", "attention"):
        bench_attention(records)
    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
