import torch
import aiop4090 as ops


def _check_gemm(fn, m, n, k, tol=1e-3, dtype=torch.float32, ref_input_dtype=None):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    out = fn(a.contiguous(), b.contiguous())
    ref_a = a.to(ref_input_dtype) if ref_input_dtype is not None else a
    ref_b = b.to(ref_input_dtype) if ref_input_dtype is not None else b
    ref = ref_a @ ref_b
    if ref.dtype != out.dtype:
        ref = ref.to(out.dtype)
    torch.cuda.synchronize()
    max_err = (out - ref).abs().max().item()
    assert max_err < tol, f"max_err={max_err}"


def test_gemm_naive_square():
    _check_gemm(ops.gemm_naive, 128, 128, 128)


def test_gemm_tiled_square():
    _check_gemm(ops.gemm_tiled, 256, 256, 256)


def test_gemm_tiled_padding_square():
    _check_gemm(ops.gemm_tiled_padding, 256, 256, 256)


def test_gemm_regtile2x2_square():
    _check_gemm(ops.gemm_regtile2x2, 256, 256, 256)


def test_gemm_regtile4x4_square():
    _check_gemm(ops.gemm_regtile4x4, 256, 256, 256)


def test_gemm_vectorized_float4_square():
    _check_gemm(ops.gemm_vectorized_float4, 256, 256, 256)


def test_gemm_wmma_fp16_square():
    _check_gemm(ops.gemm_wmma_fp16, 128, 128, 128, tol=1.0, dtype=torch.float16)


def test_gemm_wmma_fp16_float32_input():
    _check_gemm(ops.gemm_wmma_fp16, 128, 128, 128, tol=1.0, ref_input_dtype=torch.float16)


def test_gemm_non_multiple_shape():
    _check_gemm(ops.gemm_tiled, 123, 145, 67)
    _check_gemm(ops.gemm_tiled_padding, 123, 145, 67)
    _check_gemm(ops.gemm_regtile2x2, 123, 145, 67)
    _check_gemm(ops.gemm_regtile4x4, 123, 145, 67)
    _check_gemm(ops.gemm_vectorized_float4, 123, 145, 67)
    _check_gemm(ops.gemm_wmma_fp16, 123, 145, 67, tol=1.0, dtype=torch.float16)
