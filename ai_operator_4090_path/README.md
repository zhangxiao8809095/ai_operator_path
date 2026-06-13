# AI Operator 4090 Path

这是一套面向 RTX 4090 整月租用的 AI 算子开发仓库骨架。目标不是一开始超过 cuBLAS / PyTorch，而是建立完整工程闭环：

```text
CUDA kernel -> PyTorch Extension -> correctness test -> benchmark -> Nsight profiling -> 报告 -> 简历表达
```

## 0. 推荐环境

- GPU: RTX 4090 24GB
- OS: Ubuntu 22.04 / 24.04
- CUDA: 12.x
- Python: 3.10 / 3.11 / 3.12
- PyTorch: CUDA 版
- Tools: nvcc, ncu, nsys, ninja, pytest

## 1. 初始化

```bash
bash scripts/00_check_env.sh
python -m venv .venv
source .venv/bin/activate
bash scripts/10_build.sh
```

如果修改了 `.cu/.cpp` 文件：

```bash
bash scripts/clean_build.sh
bash scripts/10_build.sh
```

## 2. 正确性测试

```bash
bash scripts/20_test.sh
```

测试文件：

```text
tests/test_gemm.py
tests/test_softmax_norm.py
tests/test_attention.py
```

## 3. Benchmark

```bash
bash scripts/30_bench.sh
python benchmark/bench_gemm_shapes.py
```

## 4. Nsight Compute

```bash
bash scripts/profile_ncu.sh gemm_naive
bash scripts/profile_ncu.sh gemm_tiled
bash scripts/profile_ncu.sh gemm_regtile2x2
bash scripts/profile_ncu.sh softmax
bash scripts/profile_ncu.sh layernorm
bash scripts/profile_ncu.sh rmsnorm
bash scripts/profile_ncu.sh attention_naive
```

更完整指标：

```bash
bash scripts/profile_ncu_full.sh gemm_tiled
```

## 5. Nsight Systems

```bash
bash scripts/profile_nsys.sh
```

## 6. 阶段路径

看：

```text
docs/phase_map.md
```

## 7. 当前已包含算子

### GEMM

文件：`src/aiop4090/csrc/gemm.cu`

- `gemm_naive`
- `gemm_tiled`
- `gemm_regtile2x2`

### Softmax

文件：`src/aiop4090/csrc/softmax.cu`

- `softmax_row`

### Norm

文件：`src/aiop4090/csrc/norm.cu`

- `layernorm_row`
- `rmsnorm_row`

### Attention

文件：`src/aiop4090/csrc/attention.cu`

- `attention_naive`

## 8. 重要说明

1. 代码优先保证可读、可测、可 profiling，不是生产级最高性能。
2. `attention_naive` 故意写得直观且低效，用来观察 naive attention 的瓶颈，再引出 tiled / online softmax / FlashAttention。
3. GEMM 的后续优化路线是：padding、4x4/8x4 register tiling、vectorized load、WMMA/Tensor Core。
4. Softmax/Norm 的后续优化路线是：warp shuffle reduction、向量化加载、half/bfloat16、融合。
5. 每完成一个版本，都应该填写 `docs/report_template.md`。
