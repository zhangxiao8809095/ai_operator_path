# NCU GEMM 第五组：`gemm_naive` vs `gemm_vectorized_float4`

这份文档用于观察宽访问、A 数据复用和每线程多输出如何影响 global-memory 指令压力。`vectorized_float4` 是 naive 的独立支线，不是 regtile4x4 的后继版本。

## 本轮问题

1. 每线程计算相邻四列并使用 `float4` 读取 B 后，Duration 是否下降？
2. global load 指令和 LG Throttle 是否减少？
3. 四个累加器带来的寄存器和 occupancy 代价是否可接受？

## 代码变化

| 对比项        | `gemm_naive`          | `gemm_vectorized_float4`  |
| ------------- | --------------------- | ------------------------- |
| 每线程输出    | 1 个                  | 相邻 4 个                 |
| A 读取        | 每个输出各自读取      | 一个 A 标量供四个输出复用 |
| B 读取        | 标量 `float`          | 对齐时使用 `float4`       |
| 累加器        | 1 个                  | 4 个                      |
| block         | 16x16，共 256 threads | 16x16，共 256 threads     |
| grid.x        | `ceil(N / 16)`        | `ceil(N / 64)`            |
| shared memory | 不使用                | 不使用                    |

固定输入 `N=2048` 能够进入 `N % 4 == 0` 的 vectorized 路径。仍需在 Source/SASS 中确认编译后的实际 load 指令宽度。

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

## 运行前预测

```text
预计 Duration：
预计 LG Throttle：
预计 Registers / Thread：
预计 Achieved Occupancy：
预计 L2 / DRAM：
预测理由：
```

```text
H1：A 复用和更宽的 B load 减少 global 指令压力，LG Throttle 和 Duration 下降。
H2：寄存器、依赖链、向量地址计算或访存效率代价抵消收益。
```

## 八个关键指标记录

分别选择第 6 次稳定执行的 `gemm_naive_kernel` 和 `gemm_vectorized_float4_kernel`。

| 指标                    | NCU `Details` 中的位置                     | `gemm_naive` | `gemm_vectorized_float4` | 首轮观察问题                    |
| ----------------------- | ------------------------------------------ | ------------ | ------------------------ | ------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 待填写       | 待填写                   | vectorized 版本是否稳定更快     |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 待填写       | 待填写                   | 每线程四输出是否提高计算活跃度  |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 待填写       | 待填写                   | global/L1 指令压力是否变化      |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 待填写       | 待填写                   | 实际片外带宽是否变化            |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 待填写       | 待填写                   | A/B 请求和 cache 压力是否下降   |
| Achieved Occupancy      | `Occupancy`                                | 待填写       | 待填写                   | 四个累加器是否减少 active warps |
| Registers / Thread      | `Launch Statistics`                        | 待填写       | 待填写                   | 向量值和累加器增加多少寄存器    |
| Top Stall Reason        | `Warp State Statistics`                    | 待填写       | 待填写                   | LG Throttle 是否下降或切换      |

Top Stall 请填写：`名称：cycles，比例，Estimated Speedup`。

## 第一轮判断模板

```text
【客观变化】
Duration：
Compute / Memory：
DRAM / L2：
Occupancy / Registers：
Top Stall：

【当前能确认的 vectorization 收益】

【还不能确认的问题】
```

## 补充指标

| 补充指标                   | NCU 位置                        | `gemm_naive` | `vectorized_float4` | 用途                       |
| -------------------------- | ------------------------------- | ------------ | ------------------- | -------------------------- |
| Global Load Instructions   | Memory Tables / Source Counters | 待填写       | 待填写              | 验证动态 load 指令是否减少 |
| Global Load Requests       | Memory Tables -> L1/TEX         | 待填写       | 待填写              | 比较 L1 请求数量           |
| L1 Sectors / Request       | Memory Tables -> L1/TEX         | 待填写       | 待填写              | 检查访问合并和有效宽度     |
| L2 / DRAM Bytes            | Memory Tables                   | 待填写       | 待填写              | 区分指令减少和字节减少     |
| Eligible Warps / Scheduler | Scheduler Statistics            | 待填写       | 待填写              | 判断寄存器代价是否影响发射 |
| Local Load/Store           | Memory Tables -> Local Memory   | 待填写       | 待填写              | 检查 spill                 |

## 4090 执行命令

```bash
ITERS=1 bash scripts/profile_ncu_full.sh gemm_naive
ITERS=1 bash scripts/profile_ncu_full.sh gemm_vectorized_float4

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_naive_full.ncu-rep \
  gemm_naive_kernel 6

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_vectorized_float4_full.ncu-rep \
  gemm_vectorized_float4_kernel 6
```

## 最终结论模板

```text
1. Duration 和测试波动：
2. global load 指令/request 是否减少：
3. L1/L2/DRAM 字节和效率是否改善：
4. Registers / Occupancy 的代价：
5. LG Throttle 是否按预期下降：
6. float4 是否是当前 shape 下的有效优化：
```

## 相关文档

- [第一组：gemm_naive vs gemm_tiled](gemm_naive_vs_tiled.md)
- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)
