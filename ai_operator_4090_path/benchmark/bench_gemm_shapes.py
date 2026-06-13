import torch
import aiop4090 as ops
from bench_ops import cuda_bench

shapes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

for m, n, k in shapes:
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    for name, fn in [
        ("torch", lambda: a @ b),
        ("naive", lambda: ops.gemm_naive(a, b)),
        ("tiled", lambda: ops.gemm_tiled(a, b)),
        ("regtile2x2", lambda: ops.gemm_regtile2x2(a, b)),
    ]:
        repeat = 10 if m >= 4096 else 30
        ms = cuda_bench(fn, warmup=5, repeat=repeat)
        tflops = (2 * m * n * k) / (ms * 1e-3) / 1e12
        print(f"{m}x{k} @ {k}x{n} {name:10s}: {ms:9.3f} ms {tflops:8.3f} TFLOP/s")
