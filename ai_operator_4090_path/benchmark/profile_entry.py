import argparse
import torch
import aiop4090 as ops


def repeat(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=[
        "gemm_naive", "gemm_tiled", "gemm_tiled_padding", "gemm_regtile2x2", "gemm_regtile4x4",
        "gemm_vectorized_float4", "gemm_wmma_fp16",
        "softmax", "softmax_block_reduce", "softmax_warp_reduce", "softmax_online",
        "layernorm", "layernorm_block_reduce", "layernorm_warp_reduce", "layernorm_vectorized",
        "rmsnorm", "rmsnorm_block_reduce", "rmsnorm_warp_reduce",
        "rmsnorm_vectorized", "rmsnorm_vectorized_float4",
        "attention_naive", "attention_causal_naive", "attention_kv_cache_decode", "attention_tiled_online_softmax"
    ])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.op.startswith("gemm"):
        m = n = k = 2048
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        a_half = a.half()
        b_half = b.half()
        fn = {
            "gemm_naive": lambda: ops.gemm_naive(a, b),
            "gemm_tiled": lambda: ops.gemm_tiled(a, b),
            "gemm_tiled_padding": lambda: ops.gemm_tiled_padding(a, b),
            "gemm_regtile2x2": lambda: ops.gemm_regtile2x2(a, b),
            "gemm_regtile4x4": lambda: ops.gemm_regtile4x4(a, b),
            "gemm_vectorized_float4": lambda: ops.gemm_vectorized_float4(a, b),
            "gemm_wmma_fp16": lambda: ops.gemm_wmma_fp16(a_half, b_half),
        }[args.op]
    elif args.op == "softmax":
        x = torch.randn(8192, 4096, device="cuda")
        fn = lambda: ops.softmax_row(x)
    elif args.op == "softmax_block_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        fn = lambda: ops.softmax_block_reduce(x)
    elif args.op == "softmax_warp_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        fn = lambda: ops.softmax_warp_reduce(x)
    elif args.op == "softmax_online":
        x = torch.randn(8192, 4096, device="cuda")
        fn = lambda: ops.softmax_online(x)
    elif args.op == "layernorm":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_row(x, gamma, beta)
    elif args.op == "layernorm_warp_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_warp_reduce(x, gamma, beta)
    elif args.op == "layernorm_block_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_block_reduce(x, gamma, beta)
    elif args.op == "layernorm_vectorized":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_vectorized(x, gamma, beta)
    elif args.op == "rmsnorm":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_row(x, gamma)
    elif args.op == "rmsnorm_warp_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_warp_reduce(x, gamma)
    elif args.op == "rmsnorm_block_reduce":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_block_reduce(x, gamma)
    elif args.op == "rmsnorm_vectorized":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_vectorized(x, gamma)
    elif args.op == "rmsnorm_vectorized_float4":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_vectorized_float4(x, gamma)
    elif args.op == "attention_naive":
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: ops.attention_naive(q, k, v, True)
    elif args.op == "attention_causal_naive":
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: ops.attention_causal_naive(q, k, v)
    elif args.op == "attention_kv_cache_decode":
        q = torch.randn(1, 8, 1, 64, device="cuda")
        k_cache = torch.randn(1, 8, 128, 64, device="cuda")
        v_cache = torch.randn_like(k_cache)
        fn = lambda: ops.attention_kv_cache_decode(q, k_cache, v_cache, 128)
    elif args.op == "attention_tiled_online_softmax":
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: ops.attention_tiled_online_softmax(q, k, v, True)
    repeat(fn, args.iters)


if __name__ == "__main__":
    main()
