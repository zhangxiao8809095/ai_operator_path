import math

import pytest
import torch

import aiop4090 as ops


GEMM_FNS = [
    ops.gemm_naive,
    ops.gemm_tiled,
    ops.gemm_tiled_padding,
    ops.gemm_regtile2x2,
    ops.gemm_regtile4x4,
    ops.gemm_vectorized_float4,
]

SOFTMAX_FNS = [
    ops.softmax_row,
    ops.softmax_block_reduce,
    ops.softmax_warp_reduce,
    ops.softmax_online,
]

LAYERNORM_FNS = [
    ops.layernorm_row,
    ops.layernorm_block_reduce,
    ops.layernorm_warp_reduce,
    ops.layernorm_vectorized,
]

RMSNORM_FNS = [
    ops.rmsnorm_row,
    ops.rmsnorm_block_reduce,
    ops.rmsnorm_warp_reduce,
    ops.rmsnorm_vectorized,
    ops.rmsnorm_vectorized_float4,
]

EXPECTED_EXPORTS = [
    "gemm_naive",
    "gemm_tiled",
    "gemm_tiled_padding",
    "gemm_regtile2x2",
    "gemm_regtile4x4",
    "gemm_vectorized_float4",
    "gemm_wmma_fp16",
    "softmax_row",
    "softmax_block_reduce",
    "softmax_warp_reduce",
    "softmax_online",
    "layernorm_row",
    "layernorm_block_reduce",
    "layernorm_warp_reduce",
    "layernorm_vectorized",
    "rmsnorm_row",
    "rmsnorm_block_reduce",
    "rmsnorm_warp_reduce",
    "rmsnorm_vectorized",
    "rmsnorm_vectorized_float4",
    "attention_naive",
    "attention_causal_naive",
    "attention_kv_cache_decode",
    "attention_tiled_online_softmax",
]


def _rmsnorm_ref(x, gamma, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma


def _attention_ref(q, k, v, causal):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _misaligned_contiguous(shape, *, device="cuda"):
    numel = math.prod(shape)
    storage = torch.randn(numel + 1, device=device, dtype=torch.float32)
    tensor = storage[1:].view(shape)
    assert tensor.is_contiguous()
    assert tensor.data_ptr() % 16 != 0
    return tensor


def _print_error(label, output, reference):
    absolute = (output - reference).abs()
    max_flat_index = int(absolute.argmax().item())
    max_abs = absolute.max().item()
    max_rel = (absolute / reference.abs().clamp_min(1e-7)).max().item()
    print(
        f"{label} max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"max_flat_index={max_flat_index} nan={torch.isnan(output).sum().item()} "
        f"inf={torch.isinf(output).sum().item()}"
    )


def test_all_expected_exports_exist():
    assert all(callable(getattr(ops, name, None)) for name in EXPECTED_EXPORTS)


@pytest.mark.parametrize("fn", GEMM_FNS)
@pytest.mark.parametrize(
    ("shape", "atol", "rtol"),
    [
        ((1, 1, 1), 2e-4, 2e-4),
        ((32, 32, 32), 2e-4, 2e-4),
        ((3, 5, 7), 2e-4, 2e-4),
        ((17, 19, 33), 2e-4, 2e-4),
        ((2, 3, 4097), 2e-3, 2e-3),
    ],
)
def test_gemm_boundary_matrix(fn, shape, atol, rtol):
    m, n, k = shape
    torch.manual_seed(101)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    out = fn(a, b)
    ref = a @ b
    abs_error = (out - ref).abs()
    max_flat_index = int(abs_error.argmax().item())
    max_position = (max_flat_index // n, max_flat_index % n)
    denominator = ref.abs().clamp_min(1e-7)
    max_abs = abs_error.max().item()
    max_rel = (abs_error / denominator).max().item()

    print(
        f"{fn.__name__} shape={shape} max_abs={max_abs:.6g} "
        f"max_rel={max_rel:.6g} max_abs_position={max_position}"
    )

    assert out.shape == ref.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=atol, rtol=rtol), (
        f"shape={shape}, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, "
        f"max_abs_position={max_position}"
    )


@pytest.mark.parametrize("fn", GEMM_FNS + [ops.gemm_wmma_fp16])
@pytest.mark.parametrize("shape", [(0, 7, 5), (5, 0, 7), (5, 7, 0)])
def test_gemm_empty_dimensions(fn, shape):
    m, n, k = shape
    dtype = torch.float16 if fn is ops.gemm_wmma_fp16 else torch.float32
    a = torch.empty(m, k, device="cuda", dtype=dtype)
    b = torch.empty(k, n, device="cuda", dtype=dtype)
    out = fn(a, b)
    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    if out.numel():
        assert torch.count_nonzero(out).item() == 0
        assert torch.isfinite(out).all()
    print(
        f"{fn.__name__} empty_case={shape} output_shape={tuple(out.shape)} "
        f"dtype={out.dtype} numel={out.numel()}"
    )


@pytest.mark.parametrize("fallback", ["misaligned_b", "tail_n"])
def test_gemm_float4_misaligned_and_tail_fallback(fallback):
    torch.manual_seed(102)
    a = torch.randn(5, 7, device="cuda")
    b = (
        _misaligned_contiguous((7, 8))
        if fallback == "misaligned_b"
        else torch.randn(7, 7, device="cuda")
    )
    out = ops.gemm_vectorized_float4(a, b)
    ref = a @ b
    max_abs = (out - ref).abs().max().item()
    print(f"gemm_vectorized_float4 fallback={fallback} max_abs={max_abs:.6g}")
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("k", [16, 64, 256])
def test_gemm_wmma_error_over_k(k):
    torch.manual_seed(109)
    a = torch.randn(16, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, 16, device="cuda", dtype=torch.float16)
    out = ops.gemm_wmma_fp16(a, b)
    ref = a.float() @ b.float()
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", [(17, 19, 33), (16, 17, 32), (16, 16, 33)])
def test_gemm_wmma_error_over_k_fallback(shape):
    torch.manual_seed(110)
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = ops.gemm_wmma_fp16(a, b)
    ref = a.float() @ b.float()
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("cols", [1, 31, 32, 33, 255, 256, 257, 1024, 1026, 4096])
@pytest.mark.parametrize("fn", SOFTMAX_FNS)
def test_softmax_shape_properties(fn, cols):
    torch.manual_seed(103)
    x = torch.randn(3, cols, device="cuda") * 5.0 + 100.0
    out = fn(x)
    ref = torch.softmax(x, dim=-1)
    _print_error(f"{fn.__name__} cols={cols}", out, ref)
    assert torch.allclose(out, ref, atol=2e-5, rtol=2e-5)
    assert torch.all(out >= 0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3, device="cuda"), atol=2e-5, rtol=0)
    shifted = fn(x + 37.0)
    assert torch.allclose(out, shifted, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("fn", SOFTMAX_FNS)
def test_softmax_extreme_and_special_values(fn):
    finite = torch.tensor(
        [[1e4, 1e4 - 1, -1e4, 0.0], [-1e4, -1e4 - 1, -1e4 - 2, -1e4 - 3]],
        device="cuda",
    )
    finite_out = fn(finite)
    finite_ref = torch.softmax(finite, dim=-1)
    _print_error(f"{fn.__name__} finite_extreme", finite_out, finite_ref)
    assert torch.allclose(finite_out, finite_ref, atol=2e-5, rtol=2e-5)

    special = torch.tensor(
        [[float("nan"), 0.0, 1.0], [float("inf"), 0.0, -1.0], [float("-inf"), 0.0, 1.0]],
        device="cuda",
    )
    special_out = fn(special)
    special_ref = torch.softmax(special, dim=-1)
    print(
        f"{fn.__name__} special nan={torch.isnan(special_out).sum().item()} "
        f"inf={torch.isinf(special_out).sum().item()}"
    )
    torch.testing.assert_close(special_out, special_ref, atol=2e-5, rtol=2e-5, equal_nan=True)


@pytest.mark.parametrize("fn", SOFTMAX_FNS)
def test_softmax_repeated_execution_is_stable(fn):
    torch.manual_seed(108)
    x = torch.randn(5, 257, device="cuda")
    expected = fn(x)
    for _ in range(100):
        assert torch.equal(fn(x), expected)
    print(f"{fn.__name__} repeated_runs=100 max_run_delta=0")


@pytest.mark.parametrize("fn", SOFTMAX_FNS)
@pytest.mark.parametrize("shape", [(0, 17), (3, 0)])
def test_softmax_empty(fn, shape):
    x = torch.empty(*shape, device="cuda")
    out = fn(x)
    assert out.shape == x.shape
    assert out.numel() == 0


@pytest.mark.parametrize("fn", LAYERNORM_FNS)
@pytest.mark.parametrize("cols", [1, 31, 32, 33, 255, 256, 257, 1026])
def test_layernorm_boundary_and_eps(fn, cols):
    torch.manual_seed(104)
    x = torch.randn(3, cols, device="cuda") * 1e-2 + 10.0
    gamma = torch.randn(cols, device="cuda")
    beta = torch.randn(cols, device="cuda")
    for eps in (1e-3, 1e-5):
        out = fn(x, gamma, beta, eps)
        ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps)
        _print_error(f"{fn.__name__} cols={cols} eps={eps}", out, ref)
        assert torch.allclose(out, ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("fn", LAYERNORM_FNS)
@pytest.mark.parametrize(
    ("case", "eps"),
    [
        ("constant", 1e-3),
        ("constant", 1e-5),
        ("small_variance", 1e-5),
        ("large_offset", 1e-5),
        ("zero_eps_nonconstant", 0.0),
    ],
)
def test_layernorm_numerical_inputs(fn, case, eps):
    torch.manual_seed(112)
    cols = 257
    if case == "constant":
        x = torch.full((3, cols), 7.0, device="cuda")
    elif case == "small_variance":
        x = torch.randn(3, cols, device="cuda") * 1e-5 + 3.0
    elif case == "large_offset":
        x = torch.randn(3, cols, device="cuda") * 1e-2 + 1e3
    else:
        x = torch.randn(3, cols, device="cuda")
    gamma = torch.linspace(0.5, 1.5, cols, device="cuda")
    beta = torch.linspace(-0.25, 0.25, cols, device="cuda")
    out = fn(x, gamma, beta, eps)
    ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps)
    _print_error(f"{fn.__name__} case={case} eps={eps}", out, ref)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("fn", RMSNORM_FNS)
@pytest.mark.parametrize("cols", [1, 31, 32, 33, 258, 1026])
def test_rmsnorm_boundary_and_eps(fn, cols):
    torch.manual_seed(105)
    x = torch.randn(3, cols, device="cuda")
    gamma = torch.randn(cols, device="cuda")
    for eps in (1e-3, 1e-6):
        out = fn(x, gamma, eps)
        ref = _rmsnorm_ref(x, gamma, eps)
        _print_error(f"{fn.__name__} cols={cols} eps={eps}", out, ref)
        assert torch.allclose(out, ref, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("fn", RMSNORM_FNS)
@pytest.mark.parametrize("value", [0.0, 3.0, 1e10, 1e-10])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
def test_rmsnorm_numerical_inputs(fn, value, eps):
    x = torch.full((2, 33), value, device="cuda")
    gamma = torch.linspace(0.5, 1.5, 33, device="cuda")
    out = fn(x, gamma, eps)
    ref = _rmsnorm_ref(x, gamma, eps)
    _print_error(f"{fn.__name__} value={value} eps={eps}", out, ref)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    ("case", "cols"),
    [("misaligned", 32), ("odd_tail", 33), ("float2_only", 258), ("float4", 256)],
)
def test_norm_vectorized_misaligned_fallbacks(case, cols):
    torch.manual_seed(113)
    if case == "misaligned":
        x = _misaligned_contiguous((2, cols))
        gamma = _misaligned_contiguous((cols,))
        beta = _misaligned_contiguous((cols,))
    else:
        x = torch.randn(2, cols, device="cuda")
        gamma = torch.randn(cols, device="cuda")
        beta = torch.randn(cols, device="cuda")
    layer_out = ops.layernorm_vectorized(x, gamma, beta, 1e-5)
    layer_ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
    _print_error(f"layernorm_vectorized case={case} cols={cols}", layer_out, layer_ref)
    assert torch.allclose(layer_out, layer_ref, atol=3e-4, rtol=3e-4)
    for fn in (ops.rmsnorm_vectorized, ops.rmsnorm_vectorized_float4):
        out = fn(x, gamma, 1e-6)
        ref = _rmsnorm_ref(x, gamma, 1e-6)
        _print_error(f"{fn.__name__} case={case} cols={cols}", out, ref)
        assert torch.allclose(out, ref, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("fn", LAYERNORM_FNS)
def test_layernorm_empty(fn):
    x = torch.empty(2, 0, device="cuda")
    gamma = torch.empty(0, device="cuda")
    beta = torch.empty(0, device="cuda")
    assert fn(x, gamma, beta, 1e-5).shape == x.shape


@pytest.mark.parametrize("fn", RMSNORM_FNS)
def test_rmsnorm_empty(fn):
    x = torch.empty(0, 17, device="cuda")
    gamma = torch.empty(17, device="cuda")
    assert fn(x, gamma, 1e-6).shape == x.shape


def test_norm_rejects_invalid_eps():
    x = torch.randn(2, 8, device="cuda")
    gamma = torch.ones(8, device="cuda")
    beta = torch.zeros(8, device="cuda")
    for eps in (-1.0, float("inf"), float("nan")):
        with pytest.raises(RuntimeError, match="eps"):
            ops.layernorm_row(x, gamma, beta, eps)
        with pytest.raises(RuntimeError, match="eps"):
            ops.rmsnorm_row(x, gamma, eps)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 3, 5), (1, 2, 9, 8)])
def test_attention_boundary_shapes(causal, shape):
    torch.manual_seed(106)
    q = torch.randn(*shape, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = _attention_ref(q, k, v, causal)
    for name, fn in (
        ("attention_naive", lambda: ops.attention_naive(q, k, v, causal)),
        ("attention_tiled_online_softmax", lambda: ops.attention_tiled_online_softmax(q, k, v, causal)),
    ):
        out = fn()
        _print_error(f"{name} causal={causal} shape={shape}", out, ref)
        assert torch.allclose(out, ref, atol=4e-4, rtol=4e-4)


def test_attention_causal_mask_boundaries():
    torch.manual_seed(111)
    q = torch.randn(1, 2, 7, 5, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = _attention_ref(q, k, v, True)
    outputs = {
        "attention_naive_causal": ops.attention_naive(q, k, v, True),
        "attention_causal_naive": ops.attention_causal_naive(q, k, v),
        "attention_tiled_online_causal": ops.attention_tiled_online_softmax(q, k, v, True),
    }
    for name, out in outputs.items():
        _print_error(name, out, ref)
        assert torch.allclose(out[:, :, 0], v[:, :, 0], atol=4e-4, rtol=4e-4)
        assert torch.allclose(out[:, :, -1], ref[:, :, -1], atol=4e-4, rtol=4e-4)


@pytest.mark.parametrize("kv_len", [1, 3, 7])
def test_attention_kv_cache_boundaries(kv_len):
    torch.manual_seed(107)
    q = torch.randn(1, 2, 1, 5, device="cuda")
    k = torch.randn(1, 2, 7, 5, device="cuda")
    v = torch.randn_like(k)
    out = ops.attention_kv_cache_decode(q, k, v, kv_len)
    ref = _attention_ref(q, k[:, :, :kv_len], v[:, :, :kv_len], False)
    _print_error(f"attention_kv_cache_decode kv_len={kv_len}", out, ref)
    assert torch.allclose(out, ref, atol=4e-4, rtol=4e-4)


def test_attention_empty_batch_and_invalid_reduction_dims():
    for shape in ((0, 2, 3, 4), (1, 0, 3, 4)):
        q = torch.empty(*shape, device="cuda")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        assert ops.attention_naive(q, k, v, False).shape == q.shape
        assert ops.attention_causal_naive(q, k, v).shape == q.shape
        assert ops.attention_tiled_online_softmax(q, k, v, True).shape == q.shape

        decode_q = torch.empty(shape[0], shape[1], 1, shape[3], device="cuda")
        cache = torch.empty(shape[0], shape[1], 3, shape[3], device="cuda")
        assert ops.attention_kv_cache_decode(decode_q, cache, cache, 1).shape == decode_q.shape
        print(f"attention empty_case={shape} returned_without_launch=true")

    for shape in ((1, 1, 0, 4), (1, 1, 3, 0)):
        bad = torch.empty(*shape, device="cuda")
        with pytest.raises(RuntimeError, match="requires"):
            ops.attention_naive(bad, bad, bad, False)


def test_attention_rejects_invalid_kv_len():
    q = torch.randn(1, 1, 1, 4, device="cuda")
    k = torch.randn(1, 1, 3, 4, device="cuda")
    v = torch.randn_like(k)
    for kv_len in (0, 4):
        with pytest.raises(RuntimeError, match="kv_len"):
            ops.attention_kv_cache_decode(q, k, v, kv_len)


@pytest.mark.parametrize("op_name", EXPECTED_EXPORTS)
def test_all_exports_use_current_stream(op_name):
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    ready = torch.cuda.Event()
    completed = torch.cuda.Event()
    fn = getattr(ops, op_name)

    # Produce every input after a short delay on one non-default stream. The
    # consumer waits through an Event, then invokes the extension on a second
    # non-default stream. A clone in the same consumer stream snapshots the
    # result before the host waits, so a kernel launched on the wrong stream is
    # not made correct by the final host-side synchronization.
    with torch.cuda.stream(producer):
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(5_000_000)
        if op_name.startswith("gemm_"):
            dtype = torch.float16 if op_name == "gemm_wmma_fp16" else torch.float32
            size = 16 if op_name == "gemm_wmma_fp16" else 9
            a = torch.randn(size, size, device="cuda", dtype=dtype)
            b = torch.randn(size, size, device="cuda", dtype=dtype)
        elif op_name.startswith("softmax_"):
            x = torch.randn(5, 33, device="cuda")
        elif op_name.startswith("layernorm_"):
            x = torch.randn(5, 32, device="cuda")
            gamma = torch.randn(32, device="cuda")
            beta = torch.randn(32, device="cuda")
        elif op_name.startswith("rmsnorm_"):
            x = torch.randn(5, 32, device="cuda")
            gamma = torch.randn(32, device="cuda")
        elif op_name == "attention_kv_cache_decode":
            q = torch.randn(1, 1, 1, 8, device="cuda")
            k = torch.randn(1, 1, 5, 8, device="cuda")
            v = torch.randn_like(k)
        else:
            q = torch.randn(1, 1, 5, 8, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            causal = op_name != "attention_naive"
        ready.record()

    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        if op_name.startswith("gemm_"):
            out = fn(a, b)
        elif op_name.startswith("softmax_"):
            out = fn(x)
        elif op_name.startswith("layernorm_"):
            out = fn(x, gamma, beta, 1e-5)
        elif op_name.startswith("rmsnorm_"):
            out = fn(x, gamma, 1e-6)
        elif op_name == "attention_kv_cache_decode":
            out = fn(q, k, v, 5)
        else:
            if op_name == "attention_causal_naive":
                out = fn(q, k, v)
            else:
                out = fn(q, k, v, causal)
        observed = out.clone()
        completed.record()

    completed.synchronize()
    if op_name.startswith("gemm_"):
        ref = (a @ b).float()
    elif op_name.startswith("softmax_"):
        ref = torch.softmax(x, dim=-1)
    elif op_name.startswith("layernorm_"):
        ref = torch.nn.functional.layer_norm(x, (32,), gamma, beta, 1e-5)
    elif op_name.startswith("rmsnorm_"):
        ref = _rmsnorm_ref(x, gamma, 1e-6)
    elif op_name == "attention_kv_cache_decode":
        ref = _attention_ref(q, k, v, False)
    else:
        ref = _attention_ref(q, k, v, causal)
    tolerance = 1e-2 if op_name == "gemm_wmma_fp16" else 5e-4
    assert torch.allclose(observed, ref, atol=tolerance, rtol=tolerance)


def test_contract_rejects_cpu_wrong_dtype_and_noncontiguous():
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.softmax_row(torch.randn(2, 3))
    with pytest.raises(RuntimeError, match="float32"):
        ops.softmax_row(torch.randn(2, 3, device="cuda", dtype=torch.float16))
    noncontiguous = torch.randn(3, 4, device="cuda").transpose(0, 1)
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.softmax_row(noncontiguous)

    cuda = torch.randn(3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.gemm_naive(torch.randn(3, 4), torch.randn(4, 2))
    with pytest.raises(RuntimeError, match="float32"):
        ops.layernorm_row(cuda.half(), torch.ones(4, device="cuda").half(),
                          torch.zeros(4, device="cuda").half(), 1e-5)
    q = torch.randn(1, 1, 3, 4, device="cuda").transpose(-1, -2)
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.attention_naive(q, q, q, False)


def test_attention_rejects_invalid_contracts():
    cpu = torch.randn(1, 1, 3, 4)
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.attention_naive(cpu, cpu, cpu, False)

    q = torch.randn(1, 1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        ops.attention_naive(q.half(), q.half(), q.half(), False)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ops.attention_naive(q, torch.randn(1, 1, 4, 4, device="cuda"), q, False)
    with pytest.raises(RuntimeError, match="Q/K/V"):
        ops.attention_naive(q.squeeze(0), q.squeeze(0), q.squeeze(0), False)

    cache = torch.randn(1, 1, 5, 4, device="cuda")
    with pytest.raises(RuntimeError, match="decode length"):
        ops.attention_kv_cache_decode(q, cache, cache, 3)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ops.attention_kv_cache_decode(
            q[:, :, :1], cache, torch.randn(1, 1, 4, 4, device="cuda"), 3
        )
