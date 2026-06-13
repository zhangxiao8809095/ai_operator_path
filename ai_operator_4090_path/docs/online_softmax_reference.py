import torch


def online_softmax_vector(x: torch.Tensor, block_size: int = 16):
    """Reference online softmax for one 1D vector.

    Maintains m = running max and l = running exp-sum.
    After processing all blocks, probability = exp(x - m) / l.
    """
    m = torch.tensor(float("-inf"), device=x.device)
    l = torch.tensor(0.0, device=x.device)
    blocks = []
    for start in range(0, x.numel(), block_size):
        xb = x[start:start + block_size]
        mb = xb.max()
        lb = torch.exp(xb - mb).sum()
        m_new = torch.maximum(m, mb)
        l = torch.exp(m - m_new) * l + torch.exp(mb - m_new) * lb
        m = m_new
        blocks.append(xb)
    probs = torch.exp(x - m) / l
    return probs


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, device="cuda")
    y_online = online_softmax_vector(x, block_size=8)
    y_ref = torch.softmax(x, dim=0)
    print("max_error:", (y_online - y_ref).abs().max().item())
