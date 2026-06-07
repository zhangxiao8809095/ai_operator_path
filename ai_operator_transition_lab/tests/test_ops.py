import math
import torch
import ai_operator_lab as ops


def assert_close(a, b, tol=1e-4):
    max_err = (a - b).abs().max().item()
    assert max_err < tol, f"max_err={max_err} >= tol={tol}"


def test_gemm_naive_and_tiled():
    torch.manual_seed(0)
    A = torch.randn(128, 96, device="cuda", dtype=torch.float32)
    B = torch.randn(96, 64, device="cuda", dtype=torch.float32)
    ref = A @ B
    assert_close(ops.gemm_naive(A, B), ref, 1e-3)
    assert_close(ops.gemm_tiled(A, B), ref, 1e-3)


def test_softmax_row():
    torch.manual_seed(0)
    X = torch.randn(32, 513, device="cuda", dtype=torch.float32)
    ref = torch.softmax(X, dim=-1)
    y = ops.softmax_row(X.contiguous())
    assert_close(y, ref, 1e-5)
    assert_close(y.sum(dim=-1), torch.ones(32, device="cuda"), 1e-5)


def test_layernorm_row():
    torch.manual_seed(0)
    X = torch.randn(16, 768, device="cuda", dtype=torch.float32)
    gamma = torch.randn(768, device="cuda", dtype=torch.float32)
    beta = torch.randn(768, device="cuda", dtype=torch.float32)
    eps = 1e-5
    ref = torch.nn.functional.layer_norm(X, (768,), gamma, beta, eps)
    y = ops.layernorm_row(X.contiguous(), gamma.contiguous(), beta.contiguous(), eps)
    assert_close(y, ref, 2e-4)


def test_rmsnorm_row():
    torch.manual_seed(0)
    X = torch.randn(16, 768, device="cuda", dtype=torch.float32)
    gamma = torch.randn(768, device="cuda", dtype=torch.float32)
    eps = 1e-6
    ref = X * torch.rsqrt((X * X).mean(dim=-1, keepdim=True) + eps) * gamma
    y = ops.rmsnorm_row(X.contiguous(), gamma.contiguous(), eps)
    assert_close(y, ref, 2e-4)


def test_attention_naive_causal():
    torch.manual_seed(0)
    S, D = 64, 64
    Q = torch.randn(S, D, device="cuda", dtype=torch.float32)
    K = torch.randn(S, D, device="cuda", dtype=torch.float32)
    V = torch.randn(S, D, device="cuda", dtype=torch.float32)
    scores = (Q @ K.T) / math.sqrt(D)
    mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ V
    y = ops.attention_naive(Q.contiguous(), K.contiguous(), V.contiguous(), True)
    assert_close(y, ref, 3e-4)
