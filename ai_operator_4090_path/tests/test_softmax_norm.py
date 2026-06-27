import torch
import aiop4090 as ops


def test_softmax_row():
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    out = ops.softmax_row(x.contiguous())
    ref = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_softmax_warp_reduce():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        x = torch.randn(64, cols, device="cuda", dtype=torch.float32)
        out = ops.softmax_warp_reduce(x.contiguous())
        ref = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_softmax_online():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        x = torch.randn(64, cols, device="cuda", dtype=torch.float32)
        out = ops.softmax_online(x.contiguous())
        ref = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_layernorm_row():
    torch.manual_seed(0)
    rows, cols = 64, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
    beta = torch.randn(cols, device="cuda", dtype=torch.float32)
    out = ops.layernorm_row(x.contiguous(), gamma.contiguous(), beta.contiguous(), 1e-5)
    ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_layernorm_warp_reduce():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        rows = 64
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        beta = torch.randn(cols, device="cuda", dtype=torch.float32)
        out = ops.layernorm_warp_reduce(x.contiguous(), gamma.contiguous(), beta.contiguous(), 1e-5)
        ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_layernorm_vectorized():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        rows = 64
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        beta = torch.randn(cols, device="cuda", dtype=torch.float32)
        out = ops.layernorm_vectorized(x.contiguous(), gamma.contiguous(), beta.contiguous(), 1e-5)
        ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_rmsnorm_row():
    torch.manual_seed(0)
    rows, cols = 64, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
    out = ops.rmsnorm_row(x.contiguous(), gamma.contiguous(), 1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_rmsnorm_warp_reduce():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        rows = 64
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        out = ops.rmsnorm_warp_reduce(x.contiguous(), gamma.contiguous(), 1e-6)
        ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_rmsnorm_vectorized():
    torch.manual_seed(0)
    for cols in (1024, 1026):
        rows = 64
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
        out = ops.rmsnorm_vectorized(x.contiguous(), gamma.contiguous(), 1e-6)
        ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
