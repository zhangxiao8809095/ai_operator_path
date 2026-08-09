import argparse
import math
import torch
import aiop4090 as ops


def misaligned_contiguous(shape):
    storage = torch.randn(math.prod(shape) + 1, device="cuda")
    tensor = storage[1:].view(*shape)
    assert tensor.is_contiguous() and tensor.data_ptr() % 16 != 0
    return tensor


def repeat(fn, iters=30, label="profile"):
    torch.cuda.nvtx.range_push(label)
    try:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=[
        "gemm_naive", "gemm_tiled", "gemm_tiled_padding", "gemm_regtile2x2", "gemm_regtile4x4",
        "gemm_vectorized_float4", "gemm_vectorized_float4_misaligned", "gemm_vectorized_float4_tail",
        "gemm_wmma_fp16", "gemm_wmma_from_fp32", "gemm_wmma_fallback",
        "softmax", "softmax_row", "softmax_block_reduce", "softmax_warp_reduce", "softmax_online",
        "softmax_row_small", "softmax_block_reduce_small", "softmax_warp_reduce_small", "softmax_online_small",
        "layernorm", "layernorm_row", "layernorm_block_reduce", "layernorm_warp_reduce", "layernorm_vectorized",
        "layernorm_vectorized_misaligned", "layernorm_vectorized_tail",
        "rmsnorm", "rmsnorm_row", "rmsnorm_block_reduce", "rmsnorm_warp_reduce",
        "rmsnorm_vectorized", "rmsnorm_vectorized_float4", "rmsnorm_vectorized_float2_only",
        "rmsnorm_vectorized_float4_fallback", "rmsnorm_vectorized_float4_misaligned",
        "attention_naive", "attention_naive_noncausal", "attention_causal_naive",
        "attention_naive_s64", "attention_tiled_online_s64",
        "attention_kv_cache_decode", "attention_kv_cache_decode_kv1", "attention_kv_cache_decode_kv32",
        "attention_kv_cache_decode_kv128", "attention_kv_cache_decode_kv256",
        "attention_tiled_online_softmax", "attention_tiled_online_softmax_noncausal"
    ])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.op == "gemm_vectorized_float4_misaligned":
        m = n = k = 2048
        a = torch.randn(m, k, device="cuda")
        storage = torch.randn(k * n + 1, device="cuda")
        b = storage[1:].view(k, n)
        assert b.is_contiguous() and b.data_ptr() % 16 != 0
        fn = lambda: ops.gemm_vectorized_float4(a, b)
    elif args.op == "gemm_vectorized_float4_tail":
        m = k = 2048
        n = 2047
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        fn = lambda: ops.gemm_vectorized_float4(a, b)
    elif args.op == "gemm_wmma_from_fp32":
        m = n = k = 2048
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        fn = lambda: ops.gemm_wmma_fp16(a, b)
    elif args.op == "gemm_wmma_fallback":
        m, n, k = 2033, 2047, 2049
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        fn = lambda: ops.gemm_wmma_fp16(a, b)
    elif args.op.startswith("gemm"):
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
    elif args.op.endswith("_small") and args.op.startswith("softmax_"):
        x = torch.randn(4, 33, device="cuda")
        fn = {
            "softmax_row_small": lambda: ops.softmax_row(x),
            "softmax_block_reduce_small": lambda: ops.softmax_block_reduce(x),
            "softmax_warp_reduce_small": lambda: ops.softmax_warp_reduce(x),
            "softmax_online_small": lambda: ops.softmax_online(x),
        }[args.op]
    elif args.op in ("softmax", "softmax_row"):
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
    elif args.op == "layernorm_vectorized_misaligned":
        rows, cols = 8192, 4096
        x = misaligned_contiguous((rows, cols))
        gamma = misaligned_contiguous((cols,))
        beta = misaligned_contiguous((cols,))
        fn = lambda: ops.layernorm_vectorized(x, gamma, beta)
    elif args.op == "layernorm_vectorized_tail":
        rows, cols = 8192, 4098
        x = torch.randn(rows, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        beta = torch.randn(cols, device="cuda")
        fn = lambda: ops.layernorm_vectorized(x, gamma, beta)
    elif args.op in ("layernorm", "layernorm_row"):
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
    elif args.op == "rmsnorm_vectorized_float2_only":
        rows, cols = 8192, 4098
        x = torch.randn(rows, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        fn = lambda: ops.rmsnorm_vectorized(x, gamma)
    elif args.op == "rmsnorm_vectorized_float4_fallback":
        rows, cols = 8192, 4098
        x = torch.randn(rows, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        fn = lambda: ops.rmsnorm_vectorized_float4(x, gamma)
    elif args.op == "rmsnorm_vectorized_float4_misaligned":
        rows, cols = 8192, 4096
        x = misaligned_contiguous((rows, cols))
        gamma = misaligned_contiguous((cols,))
        fn = lambda: ops.rmsnorm_vectorized_float4(x, gamma)
    elif args.op in ("rmsnorm", "rmsnorm_row"):
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
    elif args.op in ("attention_naive_s64", "attention_tiled_online_s64"):
        q = torch.randn(1, 8, 64, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = (
            (lambda: ops.attention_naive(q, k, v, True))
            if args.op == "attention_naive_s64"
            else (lambda: ops.attention_tiled_online_softmax(q, k, v, True))
        )
    elif args.op in ("attention_naive", "attention_naive_noncausal"):
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        causal = args.op == "attention_naive"
        fn = lambda: ops.attention_naive(q, k, v, causal)
    elif args.op == "attention_causal_naive":
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: ops.attention_causal_naive(q, k, v)
    elif args.op.startswith("attention_kv_cache_decode"):
        kv_len = {
            "attention_kv_cache_decode": 128,
            "attention_kv_cache_decode_kv1": 1,
            "attention_kv_cache_decode_kv32": 32,
            "attention_kv_cache_decode_kv128": 128,
            "attention_kv_cache_decode_kv256": 256,
        }[args.op]
        q = torch.randn(1, 8, 1, 64, device="cuda")
        k_cache = torch.randn(1, 8, max(256, kv_len), 64, device="cuda")
        v_cache = torch.randn_like(k_cache)
        fn = lambda: ops.attention_kv_cache_decode(q, k_cache, v_cache, kv_len)
    elif args.op in ("attention_tiled_online_softmax", "attention_tiled_online_softmax_noncausal"):
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        causal = args.op == "attention_tiled_online_softmax"
        fn = lambda: ops.attention_tiled_online_softmax(q, k, v, causal)
    repeat(fn, args.iters, args.op)


if __name__ == "__main__":
    main()
