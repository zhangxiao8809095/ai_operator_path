# NCU GEMM 第四组：`gemm_regtile2x2` vs `gemm_regtile4x4`

这份文档用于观察更大的 register tile 是否继续提高数据复用，以及寄存器压力、occupancy 和 spill 是否开始抵消收益。八个关键指标等待从 4090 报告中填写。

## 本轮问题

1. 每线程输出从 2x2 增加到 4x4 后，Duration 是否继续下降？
2. shared load/FMA 继续下降是否带来 MIO 改善？
3. 16 个累加器是否造成寄存器、occupancy 或 local-memory spill 问题？

## 代码变化

| 对比项                       | `gemm_regtile2x2`       | `gemm_regtile4x4`       |
| ---------------------------- | ----------------------- | ----------------------- |
| 每线程输出                   | 2x2，共 4 个            | 4x4，共 16 个           |
| 累加器                       | 4 个标量                | `acc[4][4]`，共 16 个   |
| 每个 block 的输出 tile       | 32x32                   | 64x64                   |
| block                        | 16x16，共 256 threads   | 16x16，共 256 threads   |
| A/B shared memory            | `[32][16]` + `[16][32]` | `[64][16]` + `[16][64]` |
| Shared Memory / Block        | 4096 B                  | 8192 B                  |
| 每个 `kk` 的 shared load/FMA | 4 个 load / 4 个 FMA    | 8 个 load / 16 个 FMA   |

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

## 运行前预测

```text
预计 Duration：
预计 Registers / Thread：
预计 Achieved Occupancy：
预计 Local Load/Store：
预计 Top Stall Reason：
预测理由：
```

```text
H1：更高复用和 ILP 占主导，shared/MIO 压力下降，Duration 继续下降。
H2：寄存器、shared memory 或依赖代价占主导，occupancy/spill 使性能退化。
```

## 八个关键指标记录

分别选择第 6 次稳定执行的 `gemm_regtile2x2_kernel` 和 `gemm_regtile4x4_kernel`。

| 指标                    | NCU `Details` 中的位置                     | `gemm_regtile2x2` | `gemm_regtile4x4` | 首轮观察问题                           |
| ----------------------- | ------------------------------------------ | ----------------- | ----------------- | -------------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 待填写            | 待填写            | 4x4 register tile 是否更快             |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 待填写            | 待填写            | 更多 ILP 是否提高计算路径利用          |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 待填写            | 待填写            | shared/MIO 压力是否继续下降            |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 待填写            | 待填写            | 是否出现额外片外流量                   |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 待填写            | 待填写            | cache 路径是否变化                     |
| Achieved Occupancy      | `Occupancy`                                | 待填写            | 待填写            | 资源增加是否显著减少 active warps      |
| Registers / Thread      | `Launch Statistics`                        | 待填写            | 待填写            | 16 个累加器带来多少寄存器代价          |
| Top Stall Reason        | `Warp State Statistics`                    | 待填写            | 待填写            | MIO 是否下降，Scoreboard/Wait 是否上升 |

Top Stall 请填写：`名称：cycles，比例，Estimated Speedup`。

## 第一轮判断模板

```text
【客观变化】
Duration：
Compute / Memory：
DRAM / L2：
Occupancy / Registers：
Top Stall：

【收益】

【代价】

【当前支持 H1 还是 H2】
```

## 补充指标

| 补充指标                      | NCU 位置                       | `regtile2x2` | `regtile4x4` | 用途                           |
| ----------------------------- | ------------------------------ | ------------ | ------------ | ------------------------------ |
| Shared Load Instructions      | Memory Tables -> Shared Memory | 待填写       | 待填写       | 验证单位计算的 shared 指令下降 |
| Shared Load Wavefronts        | Memory Tables -> Shared Memory | 待填写       | 待填写       | 比较 MIO 实际工作量            |
| Eligible Warps / Scheduler    | Scheduler Statistics           | 待填写       | 待填写       | 判断 latency hiding 是否不足   |
| Theoretical Occupancy Limiter | Occupancy                      | 待填写       | 待填写       | 确认寄存器还是 shared 限制     |
| Local Load/Store              | Memory Tables -> Local Memory  | 待填写       | 待填写       | 检查 spill                     |
| Wait / Scoreboard Stall       | Warp State Statistics          | 待填写       | 待填写       | 检查依赖链是否加重             |

## 4090 执行命令

```bash
ITERS=1 bash scripts/profile_ncu_full.sh gemm_regtile2x2
ITERS=1 bash scripts/profile_ncu_full.sh gemm_regtile4x4

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_regtile2x2_full.ncu-rep \
  gemm_regtile2x2_kernel 6

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_regtile4x4_full.ncu-rep \
  gemm_regtile4x4_kernel 6
```

## 最终结论模板

```text
1. Duration 和波动：
2. shared load/FMA 的收益：
3. Registers / Occupancy 的代价：
4. 是否出现 local-memory spill：
5. stall 是否从 MIO 转移到依赖或其他路径：
6. 4x4 是否是当前实现的有效资源交换：
```

## 相关文档

- [第三组：gemm_tiled vs gemm_regtile2x2](gemm_tiled_vs_regtile2x2.md)
- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)
