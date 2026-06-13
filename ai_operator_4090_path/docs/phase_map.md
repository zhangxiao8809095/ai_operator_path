# 4090 整月 AI 算子开发路径

## Phase 0：环境稳定

文件：
- `scripts/00_check_env.sh`
- `scripts/10_build.sh`
- `scripts/20_test.sh`
- `scripts/30_bench.sh`

验收：
- `nvidia-smi` 能看到 RTX 4090
- `nvcc --version` 正常
- `ncu --version` 正常
- `nsys --version` 正常
- `pip install -e . --no-build-isolation` 成功
- `pytest -q tests` 成功

---

## Phase 1：GEMM 优化项目

文件：
- `src/aiop4090/csrc/gemm.cu`
- `tests/test_gemm.py`
- `benchmark/bench_gemm_shapes.py`
- `benchmark/profile_entry.py`
- `scripts/profile_ncu.sh`

当前版本：
- `gemm_naive`
- `gemm_tiled`
- `gemm_regtile2x2`

下一步 TODO：
- 增加 shared memory padding 版本
- 增加 4x4 register tiling
- 增加 vectorized load/store
- 增加 WMMA / Tensor Core 版本

重点指标：
- Duration
- SM Throughput
- Memory Throughput
- DRAM Throughput
- Achieved Occupancy
- Registers Per Thread
- Shared Memory Per Block
- Warp Stall Reasons

---

## Phase 2：Softmax / LayerNorm / RMSNorm

文件：
- `src/aiop4090/csrc/softmax.cu`
- `src/aiop4090/csrc/norm.cu`
- `tests/test_softmax_norm.py`
- `benchmark/bench_ops.py`

当前版本：
- `softmax_row`
- `layernorm_row`
- `rmsnorm_row`

下一步 TODO：
- 把 block reduction 改成 warp shuffle reduction
- softmax 加 online softmax 版本
- norm 加 half/bfloat16 版本
- RMSNorm 加向量化加载版本

重点指标：
- DRAM Throughput
- L2 Throughput
- SM Active
- Warp Stall Memory Dependency
- Launch Overhead

---

## Phase 3：mini Attention / mini FlashAttention

文件：
- `src/aiop4090/csrc/attention.cu`
- `tests/test_attention.py`
- `docs/online_softmax_reference.py`
- `docs/kv_cache_reference.py`

当前版本：
- `attention_naive`

下一步 TODO：
- 实现 tiled attention
- 实现 online softmax attention
- 增加 causal mask profiling
- 增加 KV cache reference demo

重点解释：
- naive attention 为什么产生 SxS 中间矩阵
- causal mask 如何改变可见范围
- online softmax 如何避免完整 materialize scores
- FlashAttention 为什么减少 HBM 读写

---

## Phase 4：PyTorch 自定义算子工程化

文件：
- `setup.py`
- `pyproject.toml`
- `src/aiop4090/__init__.py`
- `src/aiop4090/csrc/bindings.cpp`

验收：
- Python 可以直接调用自定义 CUDA kernel
- correctness test 对拍 PyTorch baseline
- benchmark 可以一键跑
- ncu/nsys 可以一键采样

---

## Phase 5：Nsight 报告

文件：
- `scripts/profile_ncu.sh`
- `scripts/profile_ncu_full.sh`
- `scripts/profile_nsys.sh`
- `docs/report_template.md`

建议报告：
- `01_gemm_naive_vs_tiled.md`
- `02_gemm_regtile_analysis.md`
- `03_softmax_memory_bound.md`
- `04_layernorm_vs_rmsnorm.md`
- `05_attention_naive_bottleneck.md`
- `06_4090_month_summary.md`

---

## Phase 6：简历与面试表达

文件：
- `docs/interview_resume.md`

最终交付物：
- 一个 GitHub 仓库
- 4 类算子
- PyTorch extension
- pytest correctness
- benchmark 表
- Nsight 报告
- 简历项目描述
