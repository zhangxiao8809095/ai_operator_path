import argparse
import time
import torch
import aiop4090 as ops


def cuda_bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def bench_gemm():
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
            ms = cuda_bench(fn, warmup=10, repeat=30)
            tflops = (2 * m * n * k) / (ms * 1e-3) / 1e12
            print(f"shape=({m},{n},{k}) {name:12s}: {ms:8.3f} ms, {tflops:7.3f} TFLOP/s")


def bench_softmax_norm():
    print("\n[Softmax/Norm] ms/op")
    for rows, cols in [(4096, 1024), (4096, 4096), (8192, 4096)]:
        x = torch.randn(rows, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        beta = torch.randn(cols, device="cuda")
        cases = [
            ("torch_softmax", lambda: torch.softmax(x, dim=-1)),
            ("softmax_row", lambda: ops.softmax_row(x)),
            ("softmax_warp_reduce", lambda: ops.softmax_warp_reduce(x)),
            ("softmax_online", lambda: ops.softmax_online(x)),
            ("torch_layernorm", lambda: torch.nn.functional.layer_norm(x, (cols,), gamma, beta)),
            ("layernorm_row", lambda: ops.layernorm_row(x, gamma, beta)),
            ("layernorm_warp_reduce", lambda: ops.layernorm_warp_reduce(x, gamma, beta)),
            ("layernorm_vectorized", lambda: ops.layernorm_vectorized(x, gamma, beta)),
            ("rmsnorm_row", lambda: ops.rmsnorm_row(x, gamma)),
            ("rmsnorm_warp_reduce", lambda: ops.rmsnorm_warp_reduce(x, gamma)),
            ("rmsnorm_vectorized", lambda: ops.rmsnorm_vectorized(x, gamma)),
        ]
        for name, fn in cases:
            ms = cuda_bench(fn, warmup=10, repeat=50)
            print(f"shape=({rows},{cols}) {name:16s}: {ms:8.3f} ms")


def bench_attention():
    print("\n[Attention naive] ms/op")
    # Keep shapes modest. The included naive kernel intentionally recomputes dot-products.
    for b, h, s, d in [(1, 4, 64, 64), (1, 8, 128, 64)]:
        q = torch.randn(b, h, s, d, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cases = [
            ("torch_sdp_math", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)),
            ("attention_naive", lambda: ops.attention_naive(q, k, v, True)),
        ]
        for name, fn in cases:
            ms = cuda_bench(fn, warmup=5, repeat=20)
            print(f"shape=({b},{h},{s},{d}) {name:16s}: {ms:8.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="all", choices=["all", "gemm", "softmax_norm", "attention"])
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.op in ("all", "gemm"):
        bench_gemm()
    if args.op in ("all", "softmax_norm"):
        bench_softmax_norm()
    if args.op in ("all", "attention"):
        bench_attention()


if __name__ == "__main__":
    main()
