# 简历与面试表达模板

## 简历项目名

AI Operator Kernel Optimization Lab on RTX 4090

## 简历项目描述

- 使用 CUDA C++ 实现 GEMM、Softmax、LayerNorm、RMSNorm、Naive Attention 等典型 LLM 算子。
- 完成 PyTorch C++/CUDA Extension 接入，实现 Python 侧直接调用自定义 CUDA kernel。
- 使用 pytest 对比 PyTorch baseline，验证 FP32 输出正确性。
- 使用 CUDA Event benchmark 统计 kernel 延迟，并计算 GEMM TFLOP/s。
- 使用 Nsight Compute 分析 SM Throughput、Memory Throughput、DRAM Throughput、Occupancy、Warp Stall 等指标。
- 分析 GEMM 从 naive 到 shared-memory tiled、register tiling 的性能变化。
- 分析 Softmax / LayerNorm / RMSNorm 的 memory-bound 特征。
- 实现 causal attention demo，并解释 naive attention 与 FlashAttention 在 HBM 读写上的差异。

## 面试开场表达

我原来做芯片物理层/射频软件，长期接触底层性能、硬件约束和调试问题。转向 AI 算子开发后，我用 RTX 4090 搭建了一套 CUDA 算子实验仓库，覆盖 GEMM、Softmax、Norm 和 Attention。每个算子都有 correctness test、benchmark、Nsight profiling 和报告。我重点关注的不是只把代码跑通，而是能根据 Nsight 指标判断瓶颈，并解释优化动作为什么有效。

## 高频问题

1. naive GEMM 为什么慢？
2. shared memory tiled GEMM 为什么减少 global memory 访问？
3. register tiling 为什么提升数据复用？
4. bank conflict 是怎么产生的？
5. Softmax 为什么需要 max trick？
6. online softmax 如何保证数学等价？
7. LayerNorm 和 RMSNorm 的差异是什么？
8. 为什么 Norm 类算子通常 memory-bound？
9. Attention 为什么不是单个算子，而是一条流水线？
10. FlashAttention 为什么减少 HBM 读写？
11. KV cache 为什么推理阶段有效？
12. Nsight Compute 和 Nsight Systems 分别看什么？
