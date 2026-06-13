"""AI operator optimization lab wrappers."""
from . import _C


def gemm_naive(a, b):
    return _C.gemm_naive(a, b)


def gemm_tiled(a, b):
    return _C.gemm_tiled(a, b)


def gemm_regtile2x2(a, b):
    return _C.gemm_regtile2x2(a, b)


def softmax_row(x):
    return _C.softmax_row(x)


def layernorm_row(x, gamma, beta, eps=1e-5):
    return _C.layernorm_row(x, gamma, beta, float(eps))


def rmsnorm_row(x, gamma, eps=1e-6):
    return _C.rmsnorm_row(x, gamma, float(eps))


def attention_naive(q, k, v, causal=True):
    return _C.attention_naive(q, k, v, bool(causal))
