# AI Operator 4090 Path

这是一套面向 RTX 4090 的 AI 算子开发与调试工程。目标不是一开始超过 cuBLAS / PyTorch，而是建立完整工程闭环：

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

上传后先进入本目录。租赁镜像若已经安装CUDA版PyTorch，建议让项目虚拟环境继承它；普通的空`venv`中没有PyTorch，不能直接构建扩展：

```bash
bash scripts/00_check_env.sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
bash scripts/00_check_env.sh --strict
bash scripts/10_build.sh
```

若租赁镜像没有CUDA版PyTorch，先按该服务器驱动和目标PyTorch版本安装官方CUDA wheel，再执行`10_build.sh`。不要让pip临时猜测并安装一套不匹配的CPU/CUDA PyTorch。

上传前或上传后可做不依赖GPU的文件一致性检查：

```bash
python scripts/verify_workspace.py
```

在4090服务器上，一条命令完成静态检查、环境检查、干净构建、24接口/sm_89检查和GPU冒烟测试：

```bash
bash scripts/15_verify_4090.sh smoke
# 准备进入正式实验前再运行全量正确性：
bash scripts/15_verify_4090.sh full
```

该入口会恢复脚本执行权限，因此通过zip、网页文件管理器等方式上传后也可直接用`bash`启动。若使用命令行传输，优先使用能保留目录结构和权限的`rsync -a`或`tar`。

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
tests/test_operator_validation.py
tests/test_sanitizer_smoke.py
```

完整实验顺序、停止条件、每个版本的八指标和专项指标见：

```text
docs/operator_validation_experiments_4090.md
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
bash scripts/profile_ncu.sh softmax_row
bash scripts/profile_ncu.sh softmax_block_reduce
bash scripts/profile_ncu.sh softmax_warp_reduce
bash scripts/profile_ncu.sh softmax_online
bash scripts/profile_ncu.sh layernorm_row
bash scripts/profile_ncu.sh layernorm_block_reduce
bash scripts/profile_ncu.sh layernorm_warp_reduce
bash scripts/profile_ncu.sh layernorm_vectorized
bash scripts/profile_ncu.sh rmsnorm_row
bash scripts/profile_ncu.sh rmsnorm_block_reduce
bash scripts/profile_ncu.sh rmsnorm_warp_reduce
bash scripts/profile_ncu.sh rmsnorm_vectorized
bash scripts/profile_ncu.sh rmsnorm_vectorized_float4
bash scripts/profile_ncu.sh attention_naive
bash scripts/profile_ncu.sh attention_causal_naive
bash scripts/profile_ncu.sh attention_kv_cache_decode
bash scripts/profile_ncu.sh attention_tiled_online_softmax
```

更完整指标：

```bash
bash scripts/profile_ncu_full.sh gemm_tiled
python scripts/extract_ncu_results.py --report-dir reports/ncu --family all --strict
```

NCU入口默认执行5次warmup并只保留1次正式调用（提取第6次调用），避免Full采集无意义地重复几十次。需要额外调用时显式设置`ITERS=<N>`。

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
- `layernorm_block_reduce`
- `layernorm_warp_reduce`
- `layernorm_vectorized`
- `rmsnorm_row`
- `rmsnorm_block_reduce`
- `rmsnorm_warp_reduce`
- `rmsnorm_vectorized`
- `rmsnorm_vectorized_float4`

### Attention

文件：`src/aiop4090/csrc/attention.cu`

- `attention_naive`
- `attention_causal_naive`
- `attention_kv_cache_decode`
- `attention_tiled_online_softmax`

## 8. 重要说明

1. 代码优先保证可读、可测、可 profiling，不是生产级最高性能。
2. `attention_naive` 故意写得直观且低效，用来观察naive attention的瓶颈，再与tiled online-softmax及KV-cache decode路径比较。
3. 性能采集前必须先通过正确性和目标Sanitizer闸门，不能用Profiler结果替代正确性证据。
4. `vllm_learning/`是独立学习工程，应使用它自己的虚拟环境和依赖；不要与本CUDA扩展项目共用`.venv`。
5. 每完成一个版本，都应该填写 `docs/report_template.md`。
