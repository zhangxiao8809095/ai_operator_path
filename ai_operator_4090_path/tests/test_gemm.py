import torch
import aiop4090 as ops


def _check_gemm(fn, m, n, k, tol=1e-3):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    out = fn(a.contiguous(), b.contiguous())
    ref = a @ b
    torch.cuda.synchronize()
    max_err = (out - ref).abs().max().item()
    assert max_err < tol, f"max_err={max_err}"


def test_gemm_naive_square():
    _check_gemm(ops.gemm_naive, 128, 128, 128)


def test_gemm_tiled_square():
    _check_gemm(ops.gemm_tiled, 256, 256, 256)


def test_gemm_regtile2x2_square():
    _check_gemm(ops.gemm_regtile2x2, 256, 256, 256)


def test_gemm_non_multiple_shape():
    _check_gemm(ops.gemm_tiled, 123, 145, 67)
    _check_gemm(ops.gemm_regtile2x2, 123, 145, 67)
