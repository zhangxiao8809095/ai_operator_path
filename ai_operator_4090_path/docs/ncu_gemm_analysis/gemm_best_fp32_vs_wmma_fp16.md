# NCU GEMM 第六组：最佳 FP32 版本 vs `gemm_wmma_fp16`

这份文档用于认识 Tensor Core、Compute Workload Analysis 和 Roofline。八个关键指标等待从 4090 报告中填写。

这不是严格的单变量实验：WMMA 同时改变输入 dtype、指令类型、线程协作方式和可能的 memory traffic。因此只能描述“完整方案差异”，不能把全部加速归因于 Tensor Core。

## 先确定最佳 FP32 版本

在相同 `M=N=K=2048`、相同正确性要求下运行 benchmark，从下面的 FP32 版本中选择中位 Duration 最低者：

```text
gemm_naive
gemm_tiled
gemm_tiled_padding
gemm_regtile2x2
gemm_regtile4x4
gemm_vectorized_float4
```

```text
最佳 FP32 op 名称：
最佳 FP32 kernel 名称：
benchmark 中位 Duration：
benchmark 波动范围：
```

## 方案差异

| 对比项          | 最佳 FP32 版本           | `gemm_wmma_fp16`              |
| --------------- | ------------------------ | ----------------------------- |
| 输入 dtype      | FP32                     | FP16                          |
| 累加/输出 dtype | FP32                     | FP32                          |
| 主要计算指令    | 待根据最佳版本/SASS 确认 | WMMA / Tensor Core            |
| 线程协作        | 待根据最佳版本确认       | 每个 warp 计算一个 16x16 tile |
| block           | 待填写                   | 32x4，共 128 threads          |
| 适用 shape      | 一般 FP32 路径           | M/N/K 需满足 16 的 tile 条件  |

固定的 2048 shape 能进入 WMMA 路径。`profile_entry.py` 会向 WMMA 版本传入 FP16 tensor，并获得 FP32 输出。

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

## 运行前预测

```text
预计 Duration：
预计 Compute (SM) Throughput：
预计 Memory / DRAM Throughput：
预计 Registers / Occupancy：
预计 Top Stall Reason：
预计 Tensor pipeline：
预测理由：
```

```text
H1：Tensor Core 的计算吞吐和 FP16 较低流量占主导，Duration 显著下降。
H2：数据供给、fragment load/store、并行规模或实现效率限制 Tensor Core 收益。
```

## 八个关键指标记录

分别选择第 6 次稳定执行的最佳 FP32 kernel 和 `gemm_wmma_fp16_kernel`。

| 指标                    | NCU `Details` 中的位置                     | 最佳 FP32（名称待填写） | `gemm_wmma_fp16` | 首轮观察问题                            |
| ----------------------- | ------------------------------------------ | ----------------------- | ---------------- | --------------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 待填写                  | 待填写           | WMMA 完整方案是否稳定更快               |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 待填写                  | 待填写           | 高层 SM 繁忙程度如何变化                |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 待填写                  | 待填写           | 供数路径是否成为限制                    |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 待填写                  | 待填写           | FP16 是否降低片外流量压力               |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 待填写                  | 待填写           | cache 路径是否变化                      |
| Achieved Occupancy      | `Occupancy`                                | 待填写                  | 待填写           | warp/block 配置如何影响 active warps    |
| Registers / Thread      | `Launch Statistics`                        | 待填写                  | 待填写           | fragment 带来多少寄存器分配             |
| Top Stall Reason        | `Warp State Statistics`                    | 待填写                  | 待填写           | Tensor、memory 或 dependency 路径谁主导 |

Top Stall 请填写：`名称：cycles，比例，Estimated Speedup`。

## 第一轮判断模板

```text
【比较前提】
最佳 FP32 版本：
FP32 输入/输出：
WMMA 输入/输出：
正确性 tolerance：

【八指标变化】
Duration：
Compute / Memory：
DRAM / L2：
Occupancy / Registers：
Top Stall：

【当前能确认】

【混杂变量】
```

## Tensor Core 与 Roofline 补充指标

| 补充指标                    | NCU 位置                       | 最佳 FP32 | `wmma_fp16` | 用途                                |
| --------------------------- | ------------------------------ | --------- | ----------- | ----------------------------------- |
| 实际 TFLOP/s                | 由 FLOPs / benchmark time 计算 | 待填写    | 待填写      | 比较最终有效吞吐                    |
| FP32 Pipeline Utilization   | Compute Workload Analysis      | 待填写    | 待填写      | 确认 FP32 路径压力                  |
| Tensor Pipeline Utilization | Compute Workload Analysis      | 待填写    | 待填写      | 确认 Tensor Core 是否繁忙           |
| Arithmetic Intensity        | Roofline                       | 待填写    | 待填写      | 判断 roofline 区域                  |
| 距离 Compute Roof           | Roofline                       | 待填写    | 待填写      | 判断计算吞吐利用空间                |
| DRAM / L2 Bytes             | Memory Tables                  | 待填写    | 待填写      | 分离 dtype 与 cache 流量影响        |
| Eligible Warps / Scheduler  | Scheduler Statistics           | 待填写    | 待填写      | 检查 Tensor 指令是否缺少可发射 warp |

## 4090 执行命令

先运行所有 shape 的 benchmark，并选择 2048 shape 下的最佳 FP32 版本：

```bash
python benchmark/bench_gemm_shapes.py
```

先把下面两项改成实际选出的最佳 FP32 名称：

```bash
BEST_FP32_OP=gemm_regtile4x4
BEST_FP32_KERNEL=gemm_regtile4x4_kernel

ITERS=1 bash scripts/profile_ncu_full.sh "$BEST_FP32_OP"
ITERS=1 bash scripts/profile_ncu_full.sh gemm_wmma_fp16

bash scripts/extract_ncu_metrics.sh \
  "reports/ncu/${BEST_FP32_OP}_full.ncu-rep" \
  "$BEST_FP32_KERNEL" 6

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_wmma_fp16_full.ncu-rep \
  gemm_wmma_fp16_kernel 6
```

## 最终结论模板

```text
1. 最佳 FP32 版本及选择依据：
2. 两版正确性与精度 tolerance：
3. Duration 和实际 TFLOP/s：
4. FP32 与 Tensor pipeline 利用：
5. Roofline 位置和距离上限：
6. memory traffic、Registers、Occupancy 和 stall 的代价：
7. 可以归因于什么，不能单独归因于什么：
```

## 相关文档

- [第四组：gemm_regtile2x2 vs gemm_regtile4x4](gemm_regtile2x2_vs_regtile4x4.md)
- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)
