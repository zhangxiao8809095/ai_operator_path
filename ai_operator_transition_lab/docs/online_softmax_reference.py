"""Online softmax reference for understanding FlashAttention's running max/sum update."""
import math
import torch


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    m = -float("inf")
    l = 0.0
    # First pass: update running max m and normalizer l
    for xi in x.tolist():
        m_new = max(m, xi)
        l = l * math.exp(m - m_new) + math.exp(xi - m_new)
        m = m_new
    # Second pass here only for clarity; FlashAttention also updates output accumulator online.
    return torch.exp(x - m) / l


def demo():
    x = torch.tensor([1.0, 2.0, 100.0, 3.0])
    print(online_softmax(x))
    print(torch.softmax(x, dim=0))


if __name__ == "__main__":
    demo()
