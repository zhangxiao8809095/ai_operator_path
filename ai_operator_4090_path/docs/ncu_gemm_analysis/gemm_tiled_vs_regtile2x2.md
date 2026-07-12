# NCU GEMM 第三组：`gemm_tiled` vs `gemm_regtile2x2`

这份文档用于观察 register tiling 如何改变 shared-memory 访问、寄存器使用、occupancy 和 Duration。八个关键指标已经根据 4090 报告填写。

本轮回答三个问题：

1. 每线程从 1 个输出扩展到 2x2 个输出后是否稳定变快？
2. 更多寄存器复用是否减少了每次 FMA 对 shared memory 的访问压力？
3. Registers / Thread 和 Occupancy 的代价是否值得？

## 代码变化

| 对比项                       | `gemm_tiled`            | `gemm_regtile2x2`       |
| ---------------------------- | ----------------------- | ----------------------- |
| 每线程输出                   | 1 个                    | 2x2，共 4 个            |
| 累加器                       | 1 个 `acc`              | 4 个 `acc00..acc11`     |
| 每个 block 的输出 tile       | 16x16                   | 32x32                   |
| block                        | 16x16，共 256 threads   | 16x16，共 256 threads   |
| A/B shared memory            | `[16][16]` + `[16][16]` | `[32][16]` + `[16][32]` |
| Shared Memory / Block        | 2048 B                  | 4096 B                  |
| 每个 `kk` 的 shared load/FMA | 2 个 load / 1 个 FMA    | 4 个 load / 4 个 FMA    |

`regtile2x2` 每次从 shared memory 取出 `a0/a1/b0/b1`，组合完成四次 FMA。它可能减少单位计算的 shared-load 指令，但同时增加累加器、shared memory 和每线程工作量。

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

## 运行前预测

```text
预计 Duration：
预计 Registers / Thread：
预计 Achieved Occupancy：
预计 Memory Throughput：
预计 Top Stall Reason：
预测理由：
```

保留两个竞争假设：

```text
H1：寄存器复用收益更大。
    shared load/FMA 和 MIO 压力下降，Duration 下降。

H2：资源和指令代价更大。
    Registers、shared memory、边界判断或依赖增加，Duration 不降反升。
```

## 八个关键指标记录

分别选择第 6 次稳定执行的 `gemm_tiled_kernel` 和 `gemm_regtile2x2_kernel`。

| 指标                    | NCU `Details` 中的位置                     | `gemm_tiled`                     | `gemm_regtile2x2`                | 首轮观察问题                            |
| ----------------------- | ------------------------------------------ | -------------------------------- | -------------------------------- | --------------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 2.93 ms                          | 1.07 ms                          | register tiling 是否稳定更快            |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 96.37%                           | 88.82%                           | 计算路径是否更繁忙                      |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 96.37%                           | 89.32%                           | shared/MIO 汇总压力是否变化             |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 1.33%                            | 3.31%                            | 是否仍非 DRAM 带宽瓶颈                  |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 32.95%                           | 44.57%                           | global 数据复用是否改变 L2 压力         |
| Achieved Occupancy      | `Occupancy`                                | 98.21%                           | 93.52%                           | 寄存器/shared 增加是否减少 active warps |
| Registers / Thread      | `Launch Statistics`                        | 38                               | 40                               | 四个累加器带来多少寄存器代价            |
| Top Stall Reason        | `Warp State Statistics`                    | MIO Throttle：20.4 cycles，51.5% | MIO Throttle：11.0 cycles，41.9% | MIO 是否下降或瓶颈是否迁移              |

Top Stall 请填写完整格式：

```text
名称：
cycles per issued instruction：
占 Warp Cycles Per Issued Instruction 的比例：
NCU Estimated Speedup：
```

## 八个指标逐项判断

### 1. Duration：理解正确

```text
2.93 -> 1.07 ms
时间下降 63.5%
加速约 2.74x
```

在正确性一致、CUDA Event 多轮 benchmark 也确认差异稳定的前提下，这说明 register tiling 的整体优化方向有效。对于 `2048x2048x2048` GEMM，按 NCU Duration 粗略计算的有效吞吐从约 `5.86 TFLOP/s` 提高到 `16.06 TFLOP/s`。

### 2. Compute (SM) Throughput：不能理解为有效计算下降

```text
96.37% -> 88.82%，下降 7.55 个百分点
```

`Compute (SM) Throughput` 是多个 SM 子路径组成的高层 throughput，取其中最繁忙组成部分的峰值百分比，不等于“实际 FP32 FLOP/s 达成率”。结合 tiled 的高 Memory Throughput 和 MIO Top Stall，可以怀疑片上 shared/MIO 路径是繁忙贡献者之一；regtile2x2 减少单位 FMA 的 shared load 后，最繁忙路径不再持续接近 100%，所以汇总值可以下降。具体贡献仍需查看 breakdown。

数学工作量相同而 Duration 大幅下降，实际有效 FLOP/s 反而明显提高。要确认具体哪条计算 pipeline 变化，需要查看 `Compute Workload Analysis` 中的 FP32、LSU、MIO 等 breakdown。

### 3. Memory Throughput：方向基本正确，但不是只指外部存储

```text
96.37% -> 89.32%，下降 7.05 个百分点
```

register tiling 确实增加了数据复用，但当前 `Memory Throughput` 是整个 memory 子系统的汇总值，包含 L1/TEX、shared、L2 和 DRAM，不等同于“外部存储访问”。本次最直接的变化是：

```text
tiled：      每个 kk 使用 2 次 shared load，完成 1 次 FMA
regtile2x2： 每个 kk 使用 4 次 shared load，完成 4 次 FMA

shared load / FMA：2 -> 1
```

此外，每个 block 的 global load 增加 2 倍，但输出 tile 面积增加 4 倍，所以平均每个输出对应的 global load 约减半。最终仍需使用 Shared/Global Requests 的绝对值验证。

### 4. DRAM Throughput：上升不代表总 DRAM 流量增加

```text
1.33% -> 3.31%，增加 1.98 个百分点
```

Throughput 是单位时间速率，不是总字节数。regtile2x2 的理论 global load 总量约减半，但执行时间缩短到原来的 `36.5%`。更少的数据在更短时间内完成，单位时间带宽利用率仍可能上升。

两个 DRAM 值都远低于峰值，因此当前仍不是 DRAM bandwidth-bound。要判断总片外流量是否下降，需要比较 `DRAM Bytes`，不能只比较百分比。

### 5. L2 Cache Throughput：原解释不正确

```text
32.95% -> 44.57%，增加 11.62 个百分点
```

L2 Throughput 上升不是因为使用了更多寄存器。寄存器位于 SM 内部，不会直接制造 L2 traffic。更合理的解释仍然是“速率与总量的区别”：global load 总量预计下降，但 Duration 下降得更多，L2 在单位时间内服务请求的速率反而提高。

粗略估算：如果 L2 相关工作量约减半，而时间变为 `1.07 / 2.93 = 36.5%`，单位时间速率约变为 `0.5 / 0.365 = 1.37x`；实测 L2 Throughput 约为 `44.57 / 32.95 = 1.35x`，方向一致。最终仍应比较 L2 bytes、requests 和 sectors。

### 6. Achieved Occupancy：原解释需要修改

```text
98.21% -> 93.52%，下降 4.69 个百分点
```

Occupancy 是每个 SM 上 active warps 与硬件最大 warps 的比值，不是“全 GPU 等待调度的线程总数”。两个版本的 block 都是 256 threads，因此不能直接说“每线程计算四个元素，所以每个 block 的线程减少”。

regtile2x2 的 grid block 总数确实约为 tiled 的四分之一，这可能增加尾部 wave 的不均衡；同时 Registers / Thread 从 38 增至 40、Shared Memory / Block 从 2048 B 增至 4096 B，也可能改变资源分配。但这组八指标还不能确定主要原因。

下一步应比较 `Theoretical Occupancy`、occupancy limiter、Waves Per SM 和 Eligible Warps。当前 achieved occupancy 仍超过 93%，而 Duration 大幅下降，说明这点 occupancy 代价至少没有抵消复用收益。

### 7. Registers / Thread：疑问合理，但增加 2 个并不矛盾

```text
38 -> 40，每线程增加 2 个寄存器
```

源码变量数量不等于最终寄存器分配。编译器会进行活跃区间分析和寄存器复用：四个累加器需要更长生命周期，但地址、循环和临时值可能共用物理寄存器，`a0/a1/b0/b1` 也会在使用后很快释放。

因此不能按“多 3 个累加器，就一定多 3 个寄存器”计算。40 是整个编译后 kernel 的每线程分配结果，增加 2 个是合理的。还需查看 local load/store 或 `ptxas -v`，确认编译器没有通过 spill 隐藏寄存器压力。

### 8. Top Stall Reason：名称不变是合理的，但原因不是外部内存

```text
MIO Throttle：20.4 -> 11.0 cycles，下降 46.1%
占比：        51.5% -> 41.9%，下降 9.6 个百分点
```

MIO Throttle 主要表示 MIO 指令队列繁忙，当前 GEMM 中重点对应 shared-memory 指令，不是 DRAM 或一般意义上的“外部内存”。regtile2x2 仍然需要：

- 把 A/B tile 写入 shared memory。
- 在每个 `kk` 中读取 `a0/a1/b0/b1`。
- 每个 K tile 执行两次 `__syncthreads()`。

所以 MIO 仍可能是所有等待原因中最大的一项；Top Stall 名称不必切换。关键是它的绝对等待周期和占比都明显下降，这正好支持 shared load/FMA 从 2 降到 1 的优化机制。

### 第一轮因果链

```text
每线程计算 1 个输出 -> 2x2 输出
  -> global load / output 约减半
  -> shared load / FMA 从 2 降到 1
  -> MIO Throttle cycles 下降 46.1%
  -> 付出 Registers +2、Occupancy -4.69 个百分点的代价
  -> Duration 下降 63.5%，加速约 2.74x
```

这组数据支持 H1，但仍需填写补充指标，确认 shared/global requests 的绝对数量以及是否存在 spill。

## 补充指标

| 补充指标                   | NCU 位置                       | `gemm_tiled` | `gemm_regtile2x2` | 用途                            |
| -------------------------- | ------------------------------ | ------------ | ----------------- | ------------------------------- |
| Shared Load Instructions   | Memory Tables -> Shared Memory | 待填写       | 待填写            | 验证 shared load 数量是否下降   |
| Shared Load Requests       | Memory Tables -> Shared Memory | 待填写       | 待填写            | 比较 shared 请求总工作量        |
| Shared Load Wavefronts     | Memory Tables -> Shared Memory | 待填写       | 待填写            | 比较 shared 实际处理工作量      |
| Shared Bank Conflicts      | Memory Tables -> Shared Memory | 待填写       | 待填写            | 排除新增 bank conflict          |
| Global Load Requests       | Memory Tables -> L1/TEX        | 待填写       | 待填写            | 验证 global load 总量约减半     |
| L2 / DRAM Bytes            | Memory Tables                  | 待填写       | 待填写            | 区分总流量和单位时间吞吐        |
| FMA Pipe Utilization       | Compute Workload -> 左图 FMA   | 待填写       |23.1%            | 记录 active cycles 口径的利用率 |
| Theoretical Occupancy      | Occupancy                      | 待填写       | 待填写            | 区分资源限制和 achieved 差异    |
| Waves Per SM               | Launch Statistics              | 待填写       | 待填写            | 检查 grid 缩小后的尾部效应      |
| Eligible Warps / Scheduler | Scheduler Statistics           | 待填写       | 待填写            | 判断 occupancy 下降是否影响发射 |
| Local Load/Store           | Memory Tables -> Local Memory  | 待填写       | 待填写            | 检查寄存器 spill                |

这里统一记录左侧 `Pipe Utilization (% of active cycles)` 图中的 `FMA` 数值。它包含
FP32 `FADD`、`FMUL`、`FMAD` 等指令，也可能包含 `IMUL`、`IMAD`，因此准确名称是
`FMA Pipe Utilization`，不是 NCU 中一个名为 `FP32 Pipeline Utilization` 的独立字段。

## 4090 执行命令

```bash
ITERS=1 bash scripts/profile_ncu_full.sh gemm_tiled
ITERS=1 bash scripts/profile_ncu_full.sh gemm_regtile2x2

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_tiled_full.ncu-rep \
  gemm_tiled_kernel 6

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_regtile2x2_full.ncu-rep \
  gemm_regtile2x2_kernel 6
```

## 最终结论模板

```text
1. 正确性和 benchmark 波动：
2. Duration 是否稳定变化：
3. Registers 与 Occupancy 的交换：
4. shared 指令/request/wavefront 的变化：
5. MIO 或其他 stall 的变化：
6. 代码 -> 指令/资源 -> stall -> Duration 因果链：
7. 当前结果支持 H1 还是 H2：
```

## 相关文档

- [第二组：gemm_tiled vs gemm_tiled_padding](gemm_tiled_vs_tiled_padding.md)
- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)
