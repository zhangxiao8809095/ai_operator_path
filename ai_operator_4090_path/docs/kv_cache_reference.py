import torch


def decode_without_cache(xs, w_k, w_v):
    """Recomputes all historical K/V every step. xs[t] is [D]."""
    outputs = []
    for t in range(len(xs)):
        hist = torch.stack(xs[: t + 1], dim=0)  # [T,D]
        k = hist @ w_k
        v = hist @ w_v
        outputs.append((k, v))
    return outputs


def decode_with_cache(xs, w_k, w_v):
    """Only computes current K/V and appends to cache."""
    k_cache, v_cache = [], []
    outputs = []
    for x in xs:
        k_new = x @ w_k
        v_new = x @ w_v
        k_cache.append(k_new)
        v_cache.append(v_new)
        outputs.append((torch.stack(k_cache), torch.stack(v_cache)))
    return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    D = 8
    xs = [torch.randn(D, device="cuda") for _ in range(4)]
    w_k = torch.randn(D, D, device="cuda")
    w_v = torch.randn(D, D, device="cuda")
    no_cache = decode_without_cache(xs, w_k, w_v)
    cache = decode_with_cache(xs, w_k, w_v)
    for t in range(len(xs)):
        print(t, torch.allclose(no_cache[t][0], cache[t][0]), torch.allclose(no_cache[t][1], cache[t][1]))
