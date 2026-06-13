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
        "gemm_naive", "gemm_tiled", "gemm_regtile2x2", "softmax", "layernorm", "rmsnorm", "attention_naive"
    ])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.op.startswith("gemm"):
        m = n = k = 2048
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        fn = {
            "gemm_naive": lambda: ops.gemm_naive(a, b),
            "gemm_tiled": lambda: ops.gemm_tiled(a, b),
            "gemm_regtile2x2": lambda: ops.gemm_regtile2x2(a, b),
        }[args.op]
    elif args.op == "softmax":
        x = torch.randn(8192, 4096, device="cuda")
        fn = lambda: ops.softmax_row(x)
    elif args.op == "layernorm":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        beta = torch.randn(4096, device="cuda")
        fn = lambda: ops.layernorm_row(x, gamma, beta)
    elif args.op == "rmsnorm":
        x = torch.randn(8192, 4096, device="cuda")
        gamma = torch.randn(4096, device="cuda")
        fn = lambda: ops.rmsnorm_row(x, gamma)
    elif args.op == "attention_naive":
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: ops.attention_naive(q, k, v, True)
    repeat(fn, args.iters)


if __name__ == "__main__":
    main()
