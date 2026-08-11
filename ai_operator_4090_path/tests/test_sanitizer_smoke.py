"""Small deterministic cases used by Compute Sanitizer entrypoints."""

import math

import pytest
import torch

import aiop4090 as ops


GEMM_FP32 = [
    ops.gemm_naive,
    ops.gemm_tiled,
    ops.gemm_tiled_padding,
    ops.gemm_regtile2x2,
    ops.gemm_regtile4x4,
    ops.gemm_vectorized_float4,
]
SOFTMAX = [
    ops.softmax_row,
    ops.softmax_block_reduce,
    ops.softmax_warp_reduce,
    ops.softmax_online,
]
LAYERNORM = [
    ops.layernorm_row,
    ops.layernorm_block_reduce,
    ops.layernorm_warp_reduce,
    ops.layernorm_vectorized,
]
RMSNORM = [
    ops.rmsnorm_row,
    ops.rmsnorm_block_reduce,
    ops.rmsnorm_warp_reduce,
    ops.rmsnorm_vectorized,
    ops.rmsnorm_vectorized_float4,
]


def attention_reference(q, k, v, causal):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        size = q.shape[-2]
        mask = torch.triu(torch.ones(size, size, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


@pytest.mark.parametrize("fn", GEMM_FP32)
def test_sanitizer_gemm_fp32(fn):
    torch.manual_seed(501)
    a = torch.randn(17, 19, device="cuda")
    b = torch.randn(19, 13, device="cuda")
    torch.testing.assert_close(fn(a, b), a @ b, atol=3e-4, rtol=3e-4)


def test_sanitizer_gemm_wmma():
    torch.manual_seed(502)
    a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        ops.gemm_wmma_fp16(a, b), a.float() @ b.float(), atol=5e-2, rtol=2e-2
    )


@pytest.mark.parametrize("fn", SOFTMAX)
def test_sanitizer_softmax(fn):
    torch.manual_seed(503)
    x = torch.randn(3, 33, device="cuda")
    torch.testing.assert_close(fn(x), torch.softmax(x, dim=-1), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("cols", [32, 33], ids=["vector-path", "tail-fallback"])
@pytest.mark.parametrize("fn", LAYERNORM)
def test_sanitizer_layernorm(fn, cols):
    torch.manual_seed(504)
    x = torch.randn(3, cols, device="cuda")
    gamma = torch.randn(cols, device="cuda")
    beta = torch.randn(cols, device="cuda")
    reference = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
    torch.testing.assert_close(fn(x, gamma, beta, 1e-5), reference, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("cols", [32, 33], ids=["vector-path", "tail-fallback"])
@pytest.mark.parametrize("fn", RMSNORM)
def test_sanitizer_rmsnorm(fn, cols):
    torch.manual_seed(505)
    x = torch.randn(3, cols, device="cuda")
    gamma = torch.randn(cols, device="cuda")
    reference = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * gamma
    torch.testing.assert_close(fn(x, gamma, 1e-6), reference, atol=5e-4, rtol=5e-4)


def test_sanitizer_attention():
    torch.manual_seed(506)
    q = torch.randn(1, 1, 5, 8, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    causal_reference = attention_reference(q, k, v, True)
    noncausal_reference = attention_reference(q, k, v, False)
    torch.testing.assert_close(
        ops.attention_naive(q, k, v, False), noncausal_reference, atol=5e-4, rtol=5e-4
    )
    torch.testing.assert_close(
        ops.attention_causal_naive(q, k, v), causal_reference, atol=5e-4, rtol=5e-4
    )
    torch.testing.assert_close(
        ops.attention_tiled_online_softmax(q, k, v, True),
        causal_reference,
        atol=5e-4,
        rtol=5e-4,
    )
    decode_q = q[:, :, :1, :]
    decode_reference = attention_reference(decode_q, k, v, False)
    torch.testing.assert_close(
        ops.attention_kv_cache_decode(decode_q, k, v, 5),
        decode_reference,
        atol=5e-4,
        rtol=5e-4,
    )
