"""AI operator optimization lab wrappers."""
import torch

from . import _C


def gemm_naive(a, b):
    return _C.gemm_naive(a, b)


def gemm_tiled(a, b):
    return _C.gemm_tiled(a, b)


def gemm_tiled_padding(a, b):
    return _C.gemm_tiled_padding(a, b)


def gemm_regtile2x2(a, b):
    return _C.gemm_regtile2x2(a, b)


def gemm_regtile4x4(a, b):
    return _C.gemm_regtile4x4(a, b)


def gemm_vectorized_float4(a, b):
    return _C.gemm_vectorized_float4(a, b)


def gemm_wmma_fp16(a, b):
    return _C.gemm_wmma_fp16(a, b)


def softmax_row(x):
    return _C.softmax_row(x)


def softmax_warp_reduce(x):
    return _C.softmax_warp_reduce(x)


def softmax_online(x):
    return _C.softmax_online(x)


def layernorm_row(x, gamma, beta, eps=1e-5):
    return _C.layernorm_row(x, gamma, beta, float(eps))


def layernorm_warp_reduce(x, gamma, beta, eps=1e-5):
    return _C.layernorm_warp_reduce(x, gamma, beta, float(eps))


def layernorm_vectorized(x, gamma, beta, eps=1e-5):
    return _C.layernorm_vectorized(x, gamma, beta, float(eps))


def rmsnorm_row(x, gamma, eps=1e-6):
    return _C.rmsnorm_row(x, gamma, float(eps))


def rmsnorm_warp_reduce(x, gamma, eps=1e-6):
    return _C.rmsnorm_warp_reduce(x, gamma, float(eps))


def rmsnorm_vectorized(x, gamma, eps=1e-6):
    return _C.rmsnorm_vectorized(x, gamma, float(eps))


def attention_naive(q, k, v, causal=True):
    return _C.attention_naive(q, k, v, bool(causal))
