import argparse
import torch
import ai_operator_lab as ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=["gemm_naive", "gemm_tiled", "softmax", "layernorm", "rmsnorm", "attention_naive"])
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.op.startswith("gemm"):
        A = torch.randn(2048, 2048, device="cuda")
        B = torch.randn(2048, 2048, device="cuda")
        fn = lambda: getattr(ops, args.op)(A, B)
    elif args.op == "softmax":
        X = torch.randn(4096, 4096, device="cuda")
        fn = lambda: ops.softmax_row(X)
    elif args.op == "layernorm":
        X = torch.randn(4096, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_row(X, gamma, beta, 1e-5)
    elif args.op == "rmsnorm":
        X = torch.randn(4096, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_row(X, gamma, 1e-6)
    else:
        S, D = 512, 64
        Q = torch.randn(S, D, device="cuda")
        K = torch.randn(S, D, device="cuda")
        V = torch.randn(S, D, device="cuda")
        fn = lambda: ops.attention_naive(Q, K, V, True)

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    for _ in range(args.repeat):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
