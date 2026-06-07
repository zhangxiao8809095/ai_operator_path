import math
import time
import pandas as pd
import torch
import ai_operator_lab as ops


def bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000 / repeat


def main():
    torch.manual_seed(0)
    rows = []

    for M, N, K in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        rows.append({"op": "torch_matmul", "shape": f"{M}x{K} @ {K}x{N}", "ms": bench(lambda: A @ B)})
        rows.append({"op": "gemm_naive", "shape": f"{M}x{K} @ {K}x{N}", "ms": bench(lambda: ops.gemm_naive(A, B), repeat=20)})
        rows.append({"op": "gemm_tiled", "shape": f"{M}x{K} @ {K}x{N}", "ms": bench(lambda: ops.gemm_tiled(A, B), repeat=50)})

    for R, C in [(1024, 768), (2048, 4096), (4096, 8192)]:
        X = torch.randn(R, C, device="cuda")
        gamma = torch.randn(C, device="cuda")
        beta = torch.randn(C, device="cuda")
        rows.append({"op": "torch_softmax", "shape": f"{R}x{C}", "ms": bench(lambda: torch.softmax(X, dim=-1))})
        rows.append({"op": "softmax_row", "shape": f"{R}x{C}", "ms": bench(lambda: ops.softmax_row(X))})
        rows.append({"op": "torch_layer_norm", "shape": f"{R}x{C}", "ms": bench(lambda: torch.nn.functional.layer_norm(X, (C,), gamma, beta))})
        rows.append({"op": "layernorm_row", "shape": f"{R}x{C}", "ms": bench(lambda: ops.layernorm_row(X, gamma, beta, 1e-5))})
        rows.append({"op": "rmsnorm_row", "shape": f"{R}x{C}", "ms": bench(lambda: ops.rmsnorm_row(X, gamma, 1e-6))})

    for S, D in [(128, 64), (256, 64), (512, 64)]:
        Q = torch.randn(S, D, device="cuda")
        Kt = torch.randn(S, D, device="cuda")
        V = torch.randn(S, D, device="cuda")
        rows.append({"op": "attention_naive", "shape": f"S={S},D={D},causal", "ms": bench(lambda: ops.attention_naive(Q, Kt, V, True), repeat=20)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
