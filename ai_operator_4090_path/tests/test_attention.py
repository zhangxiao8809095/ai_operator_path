import torch
import aiop4090 as ops


def _ref_attention(q, k, v, causal: bool):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    if causal:
        s = q.shape[-2]
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    prob = torch.softmax(scores, dim=-1)
    return torch.matmul(prob, v)


def test_attention_naive_causal():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = ops.attention_naive(q.contiguous(), k.contiguous(), v.contiguous(), True)
    ref = _ref_attention(q, k, v, True)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_attention_naive_non_causal():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 12, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = ops.attention_naive(q.contiguous(), k.contiguous(), v.contiguous(), False)
    ref = _ref_attention(q, k, v, False)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
