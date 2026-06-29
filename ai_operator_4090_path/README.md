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

## 3.1 GEMM 交互动画

用于从数据流和线程分工角度理解 `gemm_naive`、`gemm_tiled` 和 `gemm_regtile2x2`：

```text
docs/gemm_animation/index.html
```

## 3.2 Regtile2x2 4×4 完整动画

用于单独观察 `gemm_regtile2x2` 在一个 4×4 GEMM 上的完整执行过程：

```text
docs/regtile2x2_4x4_animation/index.html
```

内容包括：

- Global Memory 中 A/B tile 的搬运
- Shared Memory 中 As/Bs 的覆盖与复用
- 4 个线程各自的 `acc00/acc01/acc10/acc11` 寄存器变化
- 最终寄存器写回 C 的过程

## 3.3 CUDA 编程模型动画

如果对 Grid、Block、Thread、Warp、Shared Memory 等概念还不熟悉，建议先看：

```text
docs/cuda_model_animation/index.html
```

内容包括：

- Grid / Block / Thread 的层级和全局坐标计算
- Warp、SIMT、分支发散和 Block 内同步
- Register / Shared / Global Memory 的归属、生命周期和数据流
- 合并访问与跨步访问的区别

## 3.4 NCU 性能指标动画

用于理解 Nsight Compute 中常见的高层性能指标，以及如何从指标组合判断算子瓶颈：

```text
docs/ncu_metrics_animation/index.html
```

内容包括：

- Duration、Launch Stats、SM Throughput、Memory Throughput、DRAM Throughput、Achieved Occupancy、Scheduler / Warp Stall、Roofline
- NCU 报告关键行解码、metric name 命名拆解和 8 个指标的影响关系图
- Compute-bound、DRAM-bound、低 Occupancy 等典型场景
- Scheduler / Warp Stall、Roofline 和 Launch Stats 如何接到诊断动作
- 阅读 NCU 报告时的推荐诊断顺序

## 4. Nsight Compute

```bash
bash scripts/profile_ncu.sh gemm_naive
bash scripts/profile_ncu.sh gemm_tiled
bash scripts/profile_ncu.sh gemm_tiled_padding
bash scripts/profile_ncu.sh gemm_regtile2x2
bash scripts/profile_ncu.sh gemm_regtile4x4
bash scripts/profile_ncu.sh gemm_vectorized_float4
bash scripts/profile_ncu.sh gemm_wmma_fp16
bash scripts/profile_ncu.sh softmax
bash scripts/profile_ncu.sh softmax_block_reduce
bash scripts/profile_ncu.sh softmax_warp_reduce
bash scripts/profile_ncu.sh softmax_online
bash scripts/profile_ncu.sh layernorm
bash scripts/profile_ncu.sh layernorm_warp_reduce
bash scripts/profile_ncu.sh layernorm_vectorized
bash scripts/profile_ncu.sh rmsnorm
bash scripts/profile_ncu.sh rmsnorm_warp_reduce
bash scripts/profile_ncu.sh rmsnorm_vectorized
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
- `gemm_tiled_padding`
- `gemm_regtile2x2`
- `gemm_regtile4x4`
- `gemm_vectorized_float4`
- `gemm_wmma_fp16`

### Softmax

文件：`src/aiop4090/csrc/softmax.cu`

- `softmax_row`
- `softmax_block_reduce`
- `softmax_warp_reduce`
- `softmax_online`

### Norm

文件：`src/aiop4090/csrc/norm.cu`

- `layernorm_row`
- `layernorm_warp_reduce`
- `layernorm_vectorized`
- `rmsnorm_row`
- `rmsnorm_warp_reduce`
- `rmsnorm_vectorized`

### Attention

文件：`src/aiop4090/csrc/attention.cu`

- `attention_naive`

## 8. 重要说明

1. 代码优先保证可读、可测、可 profiling，不是生产级最高性能。
2. `attention_naive` 故意写得直观且低效，用来观察 naive attention 的瓶颈，再引出 tiled / online softmax / FlashAttention。
3. GEMM 的后续优化路线是：padding、4x4/8x4 register tiling、vectorized load、WMMA/Tensor Core。
4. Softmax/Norm 的后续优化路线是：warp shuffle reduction、向量化加载、half/bfloat16、融合。
5. 每完成一个版本，都应该填写 `docs/report_template.md`。
