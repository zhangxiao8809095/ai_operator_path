# AI Operator Transition Lab

This repo is a compact, debuggable CUDA/PyTorch operator lab for an AI operator/kernel engineering transition portfolio.

It contains:
- GEMM: naive and shared-memory tiled CUDA kernels
- Softmax / LayerNorm / RMSNorm: row-wise CUDA kernels
- Mini Attention: naive causal attention CUDA kernel
- PyTorch extension binding
- Correctness tests against PyTorch
- Benchmark script
- Nsight Compute / Nsight Systems profiling scripts

## Target GPU
Recommended for learning and portfolio building:
- RTX 4090 / RTX 5090 / RTX 3090 / A10 / L4 / A100
- Minimum practical VRAM: 12 GB
- Recommended VRAM: 24 GB+

## Environment
Tested design target:
- Ubuntu 22.04/24.04
- NVIDIA Driver compatible with CUDA 12.x
- CUDA Toolkit 12.x with `nvcc`
- Python 3.10-3.12
- PyTorch with CUDA support
- Nsight Compute `ncu`
- Nsight Systems `nsys`

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch pytest pandas
pip install -e .
```

## Correctness test
```bash
pytest -q tests/test_ops.py
```

## Benchmark
```bash
python benchmark/bench_ops.py
```

## Profile
```bash
bash scripts/profile_ncu.sh gemm_tiled
bash scripts/profile_ncu.sh softmax
bash scripts/profile_ncu.sh rmsnorm
bash scripts/profile_ncu.sh attention_naive

bash scripts/profile_nsys.sh
```

## Portfolio output
Each report should answer:
1. What is the baseline?
2. What changed in this kernel version?
3. What did Nsight show?
4. Which metric improved?
5. What is still slow?
6. What would be the next optimization?
