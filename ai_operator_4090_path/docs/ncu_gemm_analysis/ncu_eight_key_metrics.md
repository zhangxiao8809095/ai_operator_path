# NCU 八个关键指标详解

这份文档用于解释 GEMM 第一轮分析中使用的八个指标：

1. Time / Duration
2. Compute (SM) Throughput
3. Memory Throughput
4. DRAM Throughput
5. L2 Cache Throughput
6. Achieved Occupancy
7. Registers / Thread
8. Top Stall Reason

所有 Warp Stall 的定义、联动指标和解决思路见：[NCU Warp Stall 原因与解决思路](ncu_warp_stall_reasons.md)。

从 GEMM 版本演进学习指标并迁移到陌生 kernel 的完整流程见：[NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)。

目标不是看到一个百分比就立刻下结论，而是建立下面这条分析链：

```text
代码改动
  → GPU 执行的总工作量和指令类型发生变化
  → 各硬件路径的利用率和等待原因发生变化
  → Duration 最终发生变化
```

## 快速索引

| Metric                              | NCU Section                            | 它主要回答的问题 |
| ----------------------------------- | -------------------------------------- | ---------------- |
| Time / Duration                     | `GPU Speed Of Light Throughput`        | kernel 最终是否更快 |
| Compute (SM) Throughput             | `GPU Speed Of Light Throughput`        | SM 计算资源有多繁忙 |
| Memory Throughput                   | `GPU Speed Of Light Throughput`        | 整个 memory 子系统中最忙的路径有多繁忙 |
| DRAM Throughput                     | `GPU Speed Of Light Throughput`        | 显存带宽是否接近峰值 |
| L2 Cache Throughput                 | `GPU Speed Of Light Throughput`        | L2 数据通路是否繁忙 |
| Achieved Occupancy                  | `Occupancy`                            | 实际驻留的 active warp 是否充足 |
| Registers / Thread                  | `Launch Statistics`                    | 每线程寄存器分配是否可能限制 occupancy |
| Top Stall Reason                    | `Warp State Statistics`                | warp 最常因为什么不能发射下一条指令 |

## 1. Time / Duration

### 定义

`Duration` 是一次 kernel 从开始执行到结束所经过的时间，通常以 `us` 或 `ms` 表示。

对于完成相同计算、产生相同正确结果的两个 kernel，Duration 越低，性能越好：

```text
Speedup = baseline time / optimized time
```

当前例子：

```text
Speedup = 3.82 / 2.93 ≈ 1.30x
时间下降比例 = (3.82 - 2.93) / 3.82 ≈ 23.3%
```

### 怎么看

- Duration 是判断优化是否有效的第一指标。
- 正式性能结论应使用 NCU 外部的 CUDA Event benchmark，并进行预热和多轮重复。
- NCU 的 Duration 适合与同一份报告中的硬件指标对应，不适合单独替代稳定性 benchmark。

### 常见误区

- 只运行一次就认定变快。需要比较多轮结果的中位数和波动范围。
- 结果不正确但时间更短。必须先通过正确性测试。
- 看到某个利用率下降就否定优化。总工作量减少时，Duration 可以下降，而利用率保持不变甚至略降。

### 与其他指标的关系

其他七项指标主要用于解释 Duration 为什么变化。它们不是独立的优化目标，也不要求全部向同一个方向变化。

### Duration 增大或减小时，如何看其余七项

| Linked Metric                       | Duration 变化    | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Compute Throughput                  | 增大             | Compute 同升且接近峰值：计算 pipeline 可能受限。<br>Compute 同降：优先怀疑等待或并行度不足。 |
| Compute Throughput                  | 减小             | Compute 略降：可能只是总执行工作减少。<br>Compute 升高：计算资源利用可能改善。 |
| Memory Throughput                   | 增大             | Memory 同升：定位 L1/TEX、L2、DRAM 中最忙的层级。<br>Memory 同降：可能是发射不足。 |
| Memory Throughput                   | 减小             | Memory 不变：总 memory 工作量仍可能减少。<br>Memory 明显降低：压力可能减轻。 |
| DRAM Throughput                     | 增大             | DRAM 同时很高：怀疑显存带宽。<br>DRAM 仍低：变慢通常不是 bandwidth 导致。 |
| DRAM Throughput                     | 减小             | DRAM 同降：可能是片外流量减少。<br>DRAM 略升：可能只是执行时间缩短。 |
| L2 Throughput                       | 增大             | L2 同升：可能是 cache 请求压力增加。<br>L2 下降：检查是否转成更多 DRAM 访问。 |
| L2 Throughput                       | 减小             | L2 同降：通常与 cache 请求减少一致。<br>L2 升高：可能是供数更积极。 |
| Achieved Occupancy                  | 增大             | Occupancy 同降：检查 latency hiding。<br>Occupancy 保持很高：通常不是根因。 |
| Achieved Occupancy                  | 减小             | Occupancy 不变：优化来自其他路径。<br>Occupancy 下降：可能是寄存器复用的合理代价。 |
| Registers / Thread                  | 增大             | Registers 同增且 occupancy 降：检查寄存器限制。<br>同时排查 local-memory spill。 |
| Registers / Thread                  | 减小             | Registers 增加：可能换取数据复用。<br>Registers 减少：可能释放 occupancy。 |
| Top Stall Reason                    | 增大             | 某项 stall cycles/share 同增：它是变慢原因候选。<br>Top Stall 名称切换：说明瓶颈发生转移。 |
| Top Stall Reason                    | 减小             | 原 stall 下降或切换：可以解释优化机制。<br>stall 占比升高但绝对周期下降：不一定是退化。 |

## 2. Compute (SM) Throughput

### 定义

`Compute (SM) Throughput [%]` 表示 SM 计算相关资源达到其持续峰值的百分比。NCU 的 throughput 是由多个底层计数器汇总得到的高层指标，汇总值会突出其中最繁忙的硬件路径。

它不是下面这些概念的同义词：

- 不是 CUDA core 数量的使用比例。
- 不是 FP32 理论峰值的直接达成率。
- 不是“有效计算占总时间的比例”。

### 怎么看

- 很低：可能工作量太小、并行度不足、频繁等待或受其他资源限制。
- 很高：至少有某条 SM 计算路径很繁忙，需要进一步看 `Compute Workload Analysis` 才能知道是哪条 pipeline。
- 两个版本都很高：不能仅凭这一项判断哪个更快，应回到 Duration。

### 当前例子

```text
naive = 98.35%
tiled = 96.38%
```

两者的数学计算量相同。tiled 的 shared-memory 和同步指令改变了指令构成，因此该指标略降 1.97 个百分点，并不表示优化失败。真正结果是 Duration 下降了 23.3%。

### 常见误区

- 把 98% 理解为已经达到 RTX 4090 的 98% FP32 峰值。
- 认为 Compute Throughput 越高，kernel 就一定越快。
- 用两个百分点的变化直接判断性能，忽略 Duration 和测量波动。

### Compute Throughput 增大或减小时，如何看其余七项

| Linked Metric                       | Compute 变化     | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 下降：支持计算利用改善。<br>Duration 不降：可能只是计算 pipeline 更拥挤。 |
| Duration                            | 减小             | Duration 同降：可能是总指令减少。<br>Duration 上升：需要重点排查。 |
| Memory Throughput                   | 增大             | Memory 降：可能从 memory 转向 compute。<br>两者都高：两类路径都繁忙。 |
| Memory Throughput                   | 减小             | Memory 升：更可能偏 memory-bound。<br>两者都低：可能发射不足。 |
| DRAM Throughput                     | 增大             | DRAM 低：计算繁忙并非显存推动。<br>DRAM 高：判断双重压力。 |
| DRAM Throughput                     | 减小             | DRAM 升：显存可能限制计算发射。<br>DRAM 低：检查片上路径或同步。 |
| L2 Throughput                       | 增大             | L2 降：供数压力可能减轻。<br>L2 同升：更多 L2 供数支持计算。 |
| L2 Throughput                       | 减小             | L2 升：cache 路径可能受限。<br>L2 同降：可能总工作量减少。 |
| Achieved Occupancy                  | 增大             | occupancy 低但 Compute 高：现有 warp/ILP 已足够。 |
| Achieved Occupancy                  | 减小             | occupancy 同降：检查 latency hiding。<br>occupancy 高：通常不是 warp 数不足。 |
| Registers / Thread                  | 增大             | 寄存器增加：可能来自 register tiling。<br>确认 occupancy 代价可接受。 |
| Registers / Thread                  | 减小             | 寄存器增加且 occupancy 降：检查压力。<br>寄存器减少：排查 ILP/spill。 |
| Top Stall Reason                    | 增大             | Math Pipe Throttle 升：计算管线可能饱和。<br>memory stall 降：符合瓶颈迁移。 |
| Top Stall Reason                    | 减小             | Scoreboard/LG/MIO 升：计算可能在等待数据。<br>所有 stall 都低：检查工作量和发射规模。 |

## 3. Memory Throughput

### 定义

`Memory Throughput [%]` 是 memory 子系统的高层汇总利用率。它可能由 L1/TEX、shared memory、L2 或 DRAM 等路径中的繁忙部分驱动。

最重要的区分是：

```text
Memory Throughput 高 ≠ DRAM Throughput 高
```

例如 tiled 报告中：

```text
Memory Throughput = 96.38%
L1/TEX Throughput = 96.67%
L2 Throughput = 33.03%
DRAM Throughput = 1.32%
```

此时 memory 总指标很高，主要压力在片上的 L1/TEX/shared-memory 路径，而不是显存带宽。

### 怎么看

1. 先看 Memory Throughput 是否高。
2. 再同时看 L1/TEX、L2 和 DRAM，定位是哪一层高。
3. 最后结合 Top Stall Reason，判断繁忙是否真的让 warp 无法继续发射。

### 当前例子

```text
naive = 98.35%，Top Stall 是 LG Throttle
tiled = 96.38%，Top Stall 是 MIO Throttle
```

总利用率变化不大，但压力已经从频繁的 global/local memory 指令路径转向 shared-memory 相关的 MIO 路径。

### 常见误区

- 看到 Memory Throughput 接近 100% 就直接判断为 DRAM bandwidth-bound。
- 认为优化后 Memory Throughput 必须下降。
- 忽略 throughput 是单位时间利用率，不是总请求数或总字节数。

### Memory Throughput 增大或减小时，如何看其余七项

| Linked Metric                       | Memory 变化      | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 下降：可能是 memory 利用改善。<br>Duration 上升且 Memory 接近峰值：memory 路径可能受限。 |
| Duration                            | 减小             | Duration 同降：通常是流量/指令减少。<br>Duration 上升：可能是发射不足。 |
| Compute Throughput                  | 增大             | Compute 降：瓶颈可能偏 memory。<br>Compute 同升：计算与供数都更活跃。 |
| Compute Throughput                  | 减小             | Compute 升：可能转向 compute。<br>Compute 同降：检查 scheduler/并行度。 |
| DRAM Throughput                     | 增大             | DRAM 同升：压力可能在显存。<br>DRAM 低：压力来自片上路径。 |
| DRAM Throughput                     | 减小             | DRAM 同降：可能是片外流量减少。<br>DRAM 升：显存压力可能被总指标掩盖。 |
| L2 Throughput                       | 增大             | L2 高、DRAM 低：主要在 cache 层。<br>两者都高：后端流量大。 |
| L2 Throughput                       | 减小             | L2 同降：cache 请求可能减少。<br>L2 升：压力可能集中到 L2。 |
| Achieved Occupancy                  | 增大             | occupancy 高且 Memory 满：增加 warp 通常无效。 |
| Achieved Occupancy                  | 减小             | Occupancy 同降且 Duration 升：可能缺少并发请求和 latency hiding。 |
| Registers / Thread                  | 增大             | 寄存器降但 Memory 升：排查 spill。<br>寄存器 tiling 也可能提高供数效率。 |
| Registers / Thread                  | 减小             | 寄存器增：可能把数据留在寄存器复用。<br>同时检查 occupancy。 |
| Top Stall Reason                    | 增大             | LG/MIO/Scoreboard 升：区分队列压力与数据依赖。 |
| Top Stall Reason                    | 减小             | memory stall 同降：支持优化有效。<br>stall 不降：可能只是发射减少。 |

## 4. DRAM Throughput

### 定义

`DRAM Throughput [%]` 表示 GPU 显存数据通路达到其持续峰值带宽的百分比。它关注的是片外显存流量，不等同于源码中的 global-memory 指令数量。

global load 可能命中 cache：

```text
global load instruction
  → L1/TEX 命中：不访问 L2/DRAM
  → L2 命中：不访问 DRAM
  → L2 未命中：才需要访问 DRAM
```

### 怎么看

- DRAM 高且 Duration 受影响：可能接近显存带宽瓶颈。
- DRAM 低：说明显存带宽没有被跑满，但仍可能存在 global-memory 指令队列或访问延迟问题。
- 比较两个版本时，百分比略升不代表总 DRAM 字节数一定增加，因为执行时间也发生了变化。

### 当前例子

```text
naive = 1.09%
tiled = 1.32%
```

两个值都很低，因此当前不能把任何一个 kernel 判断为 DRAM bandwidth-bound。tiled 执行时间更短，使单位时间吞吐率略升也是可能的。

### 常见误区

- 把 DRAM Throughput 当作 global-memory 指令占比。
- 看到 DRAM 低就认为 memory 一定没有问题。LG/MIO 指令队列和 cache 路径仍可能受限。
- 只比较百分比，不比较 DRAM bytes、sectors 或 requests 等绝对工作量。

### DRAM Throughput 增大或减小时，如何看其余七项

| Linked Metric                       | DRAM 变化        | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 下降：带宽利用可能改善。<br>Duration 上升且 DRAM 接近峰值：支持 bandwidth-bound。 |
| Duration                            | 减小             | Duration 同降：支持片外流量减少。<br>Duration 上升：检查供数和并行度。 |
| Compute Throughput                  | 增大             | Compute 同升：更多数据供给计算。<br>Compute 降：计算可能在等 DRAM。 |
| Compute Throughput                  | 减小             | Compute 升：显存压力可能解除。<br>Compute 同降：问题可能转移。 |
| Memory Throughput                   | 增大             | Memory 同高：确认 DRAM 是否最高贡献者。 |
| Memory Throughput                   | 减小             | Memory 仍高：L1/L2/shared 仍繁忙。<br>Memory 同降：整体工作量可能减少。 |
| L2 Throughput                       | 增大             | L2 同升：总请求可能增加。<br>L2 降：可能更多请求穿透 L2。 |
| L2 Throughput                       | 减小             | L2 升：更多请求可能停留在 L2。<br>L2 与 DRAM 同降：总流量可能减少。 |
| Achieved Occupancy                  | 增大             | occupancy 低：可能难以隐藏 DRAM latency。 |
| Achieved Occupancy                  | 减小             | Occupancy 同降且 Duration 升：可能无法产生足够并发请求。 |
| Registers / Thread                  | 增大             | 寄存器减少：排查 spill 流量。<br>寄存器增加：检查 occupancy。 |
| Registers / Thread                  | 减小             | 寄存器增加：可能来自 register reuse。<br>强制限寄存器时仍要排除 spill。 |
| Top Stall Reason                    | 增大             | Long Scoreboard 升：偏向等待数据。<br>LG Throttle 升：偏向指令队列。 |
| Top Stall Reason                    | 减小             | Long Scoreboard 同降：支持显存等待缓解。<br>转为 MIO/Barrier：说明瓶颈迁移。 |

## 5. L2 Cache Throughput

### 定义

`L2 Cache Throughput [%]` 表示 L2 数据通路相对于其持续峰值的利用率。L2 位于所有 SM 与 DRAM 之间，可以缓存 global-memory 数据并减少 DRAM 访问。

### 怎么看

- L2 高、DRAM 低：大量请求在 L2 层被处理，显存可能不是瓶颈。
- L2 和 DRAM 都高：cache 和显存通路都可能承受较大流量。
- L2 下降：可能是请求减少，也可能只是执行时间、命中行为或其他路径发生变化，需要绝对 sectors/bytes 进一步验证。

### 当前例子

```text
naive = 37.83%
tiled = 33.03%
```

tiled 在 block 内通过 shared memory 复用 A/B，L2 压力下降 4.80 个百分点，与减少重复 cache 请求的预期一致。但 Throughput 是速率，不能单独证明 L2 总流量减少了多少。

### 常见误区

- 认为 L2 Throughput 越高越好。高可能代表命中后高效供数，也可能代表请求压力很大。
- 只看 L2 hit rate，不看请求总量。99% 命中率乘以巨大请求量仍可能产生很高压力。
- 把百分点变化当成总流量变化比例。

### L2 Throughput 增大或减小时，如何看其余七项

| Linked Metric                       | L2 变化          | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 下降：L2 供数可能更有效。<br>Duration 上升且 L2 接近峰值：cache 通路可能受限。 |
| Duration                            | 减小             | Duration 同降：通常符合请求减少。<br>Duration 上升：检查请求穿透或发射不足。 |
| Compute Throughput                  | 增大             | Compute 同升：L2 供数支持计算。<br>Compute 降：cache 压力可能阻塞计算。 |
| Compute Throughput                  | 减小             | Compute 升：L2 压力可能解除。<br>Compute 同降：可能总工作量减少。 |
| Memory Throughput                   | 增大             | Memory 同升：L2 可能是高层指标的重要贡献者。 |
| Memory Throughput                   | 减小             | Memory 仍高：压力可能转到其他 memory 层。<br>Memory 同降：整体压力减轻。 |
| DRAM Throughput                     | 增大             | DRAM 低：压力主要停留在 L2。<br>DRAM 同高：大量请求继续访问显存。 |
| DRAM Throughput                     | 减小             | DRAM 升：可能更多请求穿透 L2。<br>DRAM 同降：更符合总流量减少。 |
| Achieved Occupancy                  | 增大             | occupancy 高：通常能产生更多并发请求。<br>Duration 变差时不要继续盲目增加 warp。 |
| Achieved Occupancy                  | 减小             | Occupancy 同降且 Duration 升：L2 较低可能只是请求并发不足。 |
| Registers / Thread                  | 增大             | 寄存器降：检查数据是否退回 L2。<br>寄存器增：检查 occupancy。 |
| Registers / Thread                  | 减小             | 寄存器增：可能是 register reuse 生效。<br>寄存器降：排除 spill。 |
| Top Stall Reason                    | 增大             | Long Scoreboard/LG 同升：L2 请求可能影响发射。 |
| Top Stall Reason                    | 减小             | 对应 stall 降：支持 cache 压力解除。<br>MIO/Barrier 升：瓶颈迁移。 |

## 6. Achieved Occupancy

### 定义

Occupancy 是 SM 上实际 active warp 数量与硬件允许的最大 active warp 数量之比。`Achieved Occupancy` 是 kernel 执行期间实际观察到的 occupancy。

active warp 不代表该 warp 当前可以发射指令。一个 warp 还可能处于：

- Eligible：依赖已满足，可以发射。
- Stalled：正在等待数据、执行结果或同步。
- Issued：本周期被调度器选中并发射指令。

### 怎么看

- occupancy 太低时，可能没有足够 warp 隐藏访存和流水线延迟。
- occupancy 很高只说明 resident warp 充足，不保证这些 warp 都是 eligible。
- 应结合 `Scheduler Statistics` 中的 Eligible Warps 和 Issue 指标判断延迟是否真的被隐藏。

### 当前例子

```text
naive = 98.09%
tiled = 98.20%
```

两者基本相同。它们都使用 256 threads/block；tiled 的两个 `16x16` float shared 数组总共只有 `2048 B/block`，寄存器也没有增加，因此 occupancy 没有成为差异来源。

### 常见误区

- 把 100% occupancy 当作优化目标。更高 occupancy 不一定更快。
- occupancy 高就认为没有 stall。active warp 仍可能全部在等待。
- 为提高 occupancy 强行减少寄存器，结果发生 register spill，反而增加 local-memory 流量。

### Achieved Occupancy 增大或减小时，如何看其余七项

| Linked Metric                       | Occupancy 变化   | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 同降：更多 warp 可能改善 latency hiding。<br>Duration 不降：提高 occupancy 没有实际收益。 |
| Duration                            | 减小             | Duration 仍下降：可能换来更强复用/ILP。<br>Duration 上升：需要重点排查。 |
| Compute Throughput                  | 增大             | Compute 升：更多 eligible warp 支撑计算。<br>不变：可能已到 pipeline 上限。 |
| Compute Throughput                  | 减小             | Compute 降：可能缺少 warp 隐藏依赖。<br>Compute 高：现有 occupancy 已足够。 |
| Memory Throughput                   | 增大             | Memory 升：并发请求可能增加。<br>已接近峰值时更多 warp 未必有益。 |
| Memory Throughput                   | 减小             | Memory 降且 Duration 升：可能请求并发不足。<br>Duration 降：可能总工作量减少。 |
| DRAM Throughput                     | 增大             | DRAM 升：更多 outstanding request 利用带宽。<br>接近峰值后帮助有限。 |
| DRAM Throughput                     | 减小             | DRAM 降且 Long Scoreboard 升：可能无法隐藏显存延迟。 |
| L2 Throughput                       | 增大             | L2 升：可能有更多并发 cache 请求。<br>结合 Duration 区分供数和拥堵。 |
| L2 Throughput                       | 减小             | L2 降：可能只是请求减少。<br>Duration 升：检查供数并发。 |
| Registers / Thread                  | 增大             | 通常伴随寄存器减少或 block 改变。<br>必须排除限寄存器导致的 spill。 |
| Registers / Thread                  | 减小             | 寄存器增加是常见原因。<br>换来更低 Duration 时可以接受。 |
| Top Stall Reason                    | 增大             | Scoreboard 降：latency hiding 改善。<br>Not Selected 升：可选 warp 充足。 |
| Top Stall Reason                    | 减小             | Scoreboard 升：隐藏延迟能力下降。<br>stall 不变：occupancy 可能不关键。 |

## 7. Registers / Thread

### 定义

`Registers / Thread` 是编译器为每个线程分配的 32-bit 寄存器数量，对应 NCU 的 launch register 指标。

寄存器使用可能通过两条路径影响性能：

```text
寄存器过多
  → 每个 SM 能同时驻留的 block/warp 变少
  → occupancy 可能下降

寄存器限制过严
  → 变量溢出到 local memory
  → local load/store 增加
  → LG 或 memory latency 压力上升
```

### 怎么看

- 先比较 Registers / Thread 是否明显增加。
- 再看 Theoretical/Achieved Occupancy 是否因此下降。
- 最后检查 local-memory load/store，确认是否出现 spill。

### 当前例子

```text
naive = 40 registers/thread
tiled = 38 registers/thread
```

减少 2 个寄存器属于很小的变化，而且 occupancy 没有受到影响。shared-memory 数组本身不占每线程寄存器。具体少了哪两个寄存器不能仅凭 CUDA 源码判断，需要查看编译器生成代码和 live-register 信息。

### 常见误区

- 看到寄存器数量高就直接认定存在 register spill。
- 认为 shared-memory 数组会直接计入 Registers / Thread。
- 只追求低寄存器数量，忽略寄存器可以保存复用数据并减少访存。

### Registers / Thread 增大或减小时，如何看其余七项

| Linked Metric                       | Registers 变化   | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 下降：寄存器可能提高复用/ILP。<br>Duration 上升：检查 occupancy 损失。 |
| Duration                            | 减小             | Duration 下降：可能来自 occupancy 提升。<br>Duration 上升：检查复用损失或 spill。 |
| Compute Throughput                  | 增大             | Compute 升：可能有更多累加器和 ILP。<br>Compute 降：可能受 occupancy 限制。 |
| Compute Throughput                  | 减小             | Compute 升：可能有更多 active warp。<br>Compute 降：检查依赖和 spill。 |
| Memory Throughput                   | 增大             | Memory 降：数据可能留在寄存器复用。<br>Memory 升：仍要排查 spill。 |
| Memory Throughput                   | 减小             | Memory 升：重点排查 spill。<br>Memory 降：结合 Duration 判断。 |
| DRAM Throughput                     | 增大             | DRAM 降：可能减少片外访问。<br>DRAM 升：检查 spill 或工作集变化。 |
| DRAM Throughput                     | 减小             | DRAM/LG 同升：存在 spill 风险。<br>不能只凭寄存器减少确认。 |
| L2 Throughput                       | 增大             | L2 降：register reuse 可能生效。<br>L2 升：可能复用变差。 |
| L2 Throughput                       | 减小             | L2 升：检查数据是否退回 memory 层级。 |
| Achieved Occupancy                  | 增大             | 下降是常见代价。<br>Duration 更低且 warp 足够时可接受。 |
| Achieved Occupancy                  | 减小             | 上升是常见收益。<br>性能不升说明原 occupancy 可能已足够。 |
| Top Stall Reason                    | 增大             | Dependency/Math stall 降：ILP 可能改善。<br>Scoreboard 升：occupancy 可能过低。 |
| Top Stall Reason                    | 减小             | LG/Long Scoreboard 升：检查 spill。<br>Not Selected 升：可调度 warp 可能更多。 |

## 8. Top Stall Reason

### 定义

warp stall 表示 warp 为什么暂时不能发射下一条指令。`Warp State Statistics` 使用 `cycles per issued instruction` 表示各状态平均占用的周期。

确定 Top Stall Reason 的基本过程：

1. 读取 `Warp Cycles Per Issued Instruction`，得到总指令间隔。
2. 比较各 stall 的 `cycles per issued instruction`。
3. 记录数值最大的一项及其占比。
4. 回到 Source/SASS，寻找产生该等待的指令位置。

NCU 下方显示的优化提示由规则触发，不是固定的 Top Stall 列。如果出现多项提示，仍要比较各自周期数。

### GEMM 中常见的 stall

| Stall                         | 通俗解释 |
| ----------------------------- | -------- |
| LG Throttle                   | local/global memory 指令队列已满，常见于过于频繁地发射 global/local load/store |
| MIO Throttle                  | MIO 指令队列已满，可能来自 shared-memory、特殊数学或动态分支指令压力 |
| Long Scoreboard               | 等待 global/local/texture 等较长延迟的数据依赖返回 |
| Short Scoreboard              | 等待 shared-memory 或部分较短延迟的数据依赖 |
| Barrier                       | 等待同一 block 中其他 warp 到达同步点 |
| Not Selected                  | warp 已经 eligible，但调度器本周期选择了其他 warp；数值高不一定是坏事 |

### 当前例子：naive

```text
Warp Cycles Per Issued Instruction = 41.09 cycles
LG Throttle                        = 27.0 cycles
占比                               = 27.0 / 41.09 ≈ 65.6%
```

代码在每次 K 循环中都读取 A/B global memory，因此 LG 指令队列成为主要压力。

### 当前例子：tiled

```text
Warp Cycles Per Issued Instruction = 39.54 cycles
MIO Throttle                       = 20.4 cycles
占比                               = 20.4 / 39.54 ≈ 51.5%
```

tiled 将大部分数据复用转移到 shared memory，LG 压力下降，但反复 shared-memory 读取让 MIO 成为新的主要等待。

### 常见误区

- 把 Top Stall Reason 当成整个 kernel 唯一的性能瓶颈。
- 把 stall 占比当作可获得的加速比例。
- 把 `Est. Speedup` 当作 stall 占比。它只是 NCU 规则估计的局部优化上限。
- 看到 stall 很高就立即改代码。只有调度器无法持续发射时，stall 才一定值得优先处理。

### Top Stall 增大、减小或切换时，如何看其余七项

这里的“增大/减小”应比较同一种 stall 的 `cycles per issued instruction` 和占比；如果名称发生变化，则表示主要等待路径发生了切换。

| Linked Metric                       | Top Stall 变化   | 如何联动判断 |
| ----------------------------------- | ---------------- | ------------ |
| Duration                            | 增大             | Duration 同升：该 stall 更可能造成退化。<br>Duration 降：可能是其他状态减少更多。 |
| Duration                            | 减小/切换        | Duration 同降：支持优化有效。<br>Duration 不变：可能出现新瓶颈。 |
| Compute Throughput                  | 增大             | Compute 降且 memory stall 升：计算在等数据。<br>Math Pipe 与 Compute 同升：计算管线可能饱和。 |
| Compute Throughput                  | 减小/切换        | Compute 升：等待解除。<br>Compute 不变：其他资源可能接棒。 |
| Memory Throughput                   | 增大             | Top Stall 是 LG/MIO/Scoreboard 且 Memory 高：定位 memory 路径。<br>Top Stall 是 Barrier：不要归因于 Memory。 |
| Memory Throughput                   | 减小/切换        | 切换到 Math Pipe/Barrier：瓶颈迁移。<br>Memory 不降：可能只是片上层级改变。 |
| DRAM Throughput                     | 增大             | Long Scoreboard 与 DRAM 同高：偏显存等待。<br>LG 高但 DRAM 低：偏指令队列。 |
| DRAM Throughput                     | 减小/切换        | 若下降的是 Long Scoreboard 且 DRAM 同降：支持片外等待缓解。<br>DRAM 不降：原 stall 可能不是显存带宽导致。 |
| L2 Throughput                       | 增大             | Long Scoreboard/LG 与 L2 同升：检查 cache。<br>MIO 高：更偏 shared 路径。 |
| L2 Throughput                       | 减小/切换        | 若下降的是 LG/Long Scoreboard 且 L2 同降：支持 cache 压力减轻。<br>切换为 MIO：压力转向 shared。 |
| Achieved Occupancy                  | 增大             | occupancy 低：stall 更难隐藏。<br>occupancy 高仍 stall：active 不等于 eligible。 |
| Achieved Occupancy                  | 减小/切换        | occupancy 不变且 stall 降：直接改善依赖/队列。<br>occupancy 降但 Duration 降：可以接受。 |
| Registers / Thread                  | 增大             | 寄存器增、occupancy 降、Scoreboard 升：检查压力。<br>LG 升：排查 spill。 |
| Registers / Thread                  | 减小/切换        | Registers 增加且 memory stall 下降：可能是寄存器复用生效。<br>Registers 降且 LG 升：排查 spill。 |

## 八个指标如何一起使用

建议固定按下面的顺序分析：

```text
1. Duration：改动是否稳定变快？
2. Compute vs Memory：当前繁忙方向偏计算还是偏 memory？
3. DRAM vs L2：memory 压力位于显存还是 cache？
4. Occupancy + Registers：是否缺少足够 active warp 隐藏延迟？
5. Top Stall：warp 具体在等待什么？
6. Source/SASS：是哪一行代码产生了这些指令或等待？
```

判断一次优化有效，至少需要三层证据：

| Evidence                      | 需要看到什么 |
| ----------------------------- | ------------ |
| 结果正确                      | 与 PyTorch baseline 对拍通过 |
| 性能结果                      | 多轮 benchmark 中 Duration 稳定下降，差异明显大于测试波动 |
| 硬件因果                      | 绝对指令/请求数量和 stall 变化符合代码改动的预期 |

利用率百分比可以不明显变化。只要完成相同正确工作所需的总时间稳定下降，并且硬件行为与代码改动相符，优化就是有效的。

## 官方参考

- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Nsight Compute UI Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
