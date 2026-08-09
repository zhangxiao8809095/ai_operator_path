import torch
import aiop4090 as ops
from bench_ops import cuda_bench, format_stats

shapes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

for m, n, k in shapes:
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    a_half = a.half()
    b_half = b.half()
    for name, fn in [
        ("torch", lambda: a @ b),
        ("naive", lambda: ops.gemm_naive(a, b)),
        ("tiled", lambda: ops.gemm_tiled(a, b)),
        ("tiled_pad", lambda: ops.gemm_tiled_padding(a, b)),
        ("regtile2x2", lambda: ops.gemm_regtile2x2(a, b)),
        ("regtile4x4", lambda: ops.gemm_regtile4x4(a, b)),
        ("float4", lambda: ops.gemm_vectorized_float4(a, b)),
        ("wmma_fp16", lambda: ops.gemm_wmma_fp16(a_half, b_half)),
    ]:
        repeat = 10 if m >= 4096 else 30
        stats = cuda_bench(fn, warmup=5, repeat=repeat)
        median_ms = stats["median"]
        tflops = (2 * m * n * k) / (median_ms * 1e-3) / 1e12
        print(
            f"{m}x{k} @ {k}x{n} {name:10s}: "
            f"{format_stats(stats)} {tflops:8.3f} TFLOP/s"
        )
