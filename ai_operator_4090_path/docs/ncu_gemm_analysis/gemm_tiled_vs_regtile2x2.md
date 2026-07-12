# NCU GEMM 第三组：`gemm_tiled` vs `gemm_regtile2x2`

这份文档用于观察 register tiling 如何改变 shared-memory 访问、寄存器使用、occupancy 和 Duration。八个关键指标全部等待从 4090 报告中填写。

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

| 指标                    | NCU `Details` 中的位置                     | `gemm_tiled` | `gemm_regtile2x2` | 首轮观察问题                            |
| ----------------------- | ------------------------------------------ | ------------ | ----------------- | --------------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 待填写       | 待填写            | register tiling 是否稳定更快            |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 待填写       | 待填写            | 计算路径是否更繁忙                      |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 待填写       | 待填写            | shared/MIO 汇总压力是否变化             |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 待填写       | 待填写            | 是否仍非 DRAM 带宽瓶颈                  |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 待填写       | 待填写            | global 数据复用是否改变 L2 压力         |
| Achieved Occupancy      | `Occupancy`                                | 待填写       | 待填写            | 寄存器/shared 增加是否减少 active warps |
| Registers / Thread      | `Launch Statistics`                        | 待填写       | 待填写            | 四个累加器带来多少寄存器代价            |
| Top Stall Reason        | `Warp State Statistics`                    | 待填写       | 待填写            | MIO 是否下降或瓶颈是否迁移              |

Top Stall 请填写完整格式：

```text
名称：
cycles per issued instruction：
占 Warp Cycles Per Issued Instruction 的比例：
NCU Estimated Speedup：
```

## 第一轮判断模板

```text
【客观变化】
Duration：
Compute / Memory：
DRAM / L2：
Occupancy / Registers：
Top Stall：

【当前假设】

【还缺少的证据】
```

## 补充指标

| 补充指标                   | NCU 位置                       | `gemm_tiled` | `gemm_regtile2x2` | 用途                            |
| -------------------------- | ------------------------------ | ------------ | ----------------- | ------------------------------- |
| Shared Load Instructions   | Memory Tables -> Shared Memory | 待填写       | 待填写            | 验证 shared load 数量是否下降   |
| Shared Load Requests       | Memory Tables -> Shared Memory | 待填写       | 待填写            | 比较 shared 请求总工作量        |
| Shared Load Wavefronts     | Memory Tables -> Shared Memory | 待填写       | 待填写            | 比较 shared 实际处理工作量      |
| Shared Bank Conflicts      | Memory Tables -> Shared Memory | 待填写       | 待填写            | 排除新增 bank conflict          |
| Eligible Warps / Scheduler | Scheduler Statistics           | 待填写       | 待填写            | 判断 occupancy 下降是否影响发射 |
| Local Load/Store           | Memory Tables -> Local Memory  | 待填写       | 待填写            | 检查寄存器 spill                |

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
