# NCU Warp Stall 原因与解决思路

这份文档用于在 RTX 4090 上分析 Nsight Compute 的 `Warp State Statistics`。

目标不是看到最大的 stall 就立即修改代码，而是回答三个问题：

```text
1. warp 在等待什么？
2. 是哪类指令或代码造成的？
3. 这个等待是否真的让 kernel 变慢？
```

## Stall 是什么

一个 warp 只有在下一条指令已经满足条件时，才是 `Eligible Warp`。调度器从 eligible warps 中选择 warp 发射指令：

```text
Active Warp
  -> 条件不满足：归入某种 stall 状态
  -> 条件已满足：成为 Eligible Warp
       -> 被调度器选择：Selected，发射指令
       -> 没被选择：Not Selected
```

`Warp State Statistics` 中的 `cycles per issued instruction` 表示：平均每发射一条 warp 指令，各状态分别占用了多少 warp cycles。

需要先记住：

- stall 是 warp 级等待，不等于整个 GPU 在同一时间空闲。
- 一个 warp 等待时，调度器可能发射其他 warp，因此 stall 可能被隐藏。
- `Top Stall Reason` 是占比最大的状态，不一定是性能收益最大的优化点。
- `Selected` 是成功发射指令的正常状态，不是 stall。
- 只有 scheduler 不能持续发射时，stall 才应成为主要排查方向。

## 4090 上有多少种

RTX 4090 属于 Ada 架构。常见 NCU 报告包含 **17 种 stall reason，加上 1 种正常的 `Selected` 状态**。

不同 CUDA/NCU 版本可能调整显示名称或可用指标。应在 4090 服务器上用下面的命令确认当前版本：

```bash
ncu --query-metrics-mode all | grep 'smsp__warp_issue_stalled_'
```

新版 NCU 文档还包含 `Warpgroup Arrive`。它与新架构上的 warpgroup 指令有关，通常不适用于 RTX 4090，本文放在最后单独说明。

## 快速索引

| Stall Reason                       | warp 正在等待什么                         | 优先排查方向                                 |
| ---------------------------------- | ----------------------------------------- | -------------------------------------------- |
| Barrier                            | 同一 block 的其他 warp 到达 CTA barrier   | `__syncthreads()`、到达同步点前的负载不均衡  |
| Branch Resolving                   | 分支目标和程序计数器更新                  | 频繁分支、复杂控制流、warp divergence        |
| Dispatch Stall                     | 已准备好的指令被 dispatcher 暂缓          | Source/SASS、指令组合、硬件冲突              |
| Drain                              | `EXIT` 后尚未完成的 memory 操作           | kernel 尾部大量 store、写访问效率            |
| IMC Miss                           | immediate constant cache miss             | warp 内 constant 地址不统一、常量访问局部性  |
| LG Throttle                        | local/global memory 指令队列腾出空间      | global/local 指令过多、寄存器 spill          |
| Long Scoreboard                    | L1TEX 的 global/local/texture 数据返回    | 访存延迟、合并访问、cache 命中、数据依赖     |
| Math Pipe Throttle                 | 某条数学执行流水线腾出空间                | 特定数学 pipeline 过度集中                   |
| Membar                             | memory barrier/fence 完成                 | 不必要的 fence、尚未完成的 memory 操作       |
| MIO Throttle                       | MIO 指令队列腾出空间                      | shared memory、特殊数学、动态分支指令过多    |
| Misc                               | 未单独分类的硬件原因                      | Source/SASS、其他 section、工具版本          |
| No Instructions                    | 指令获取或 instruction cache              | kernel 太短、grid 太小、代码体积和跳转过大   |
| Not Selected                       | 调度器选择其他 eligible warp              | 通常说明可调度 warp 充足，不一定需要解决     |
| Short Scoreboard                   | MIO 数据依赖完成                          | shared memory、bank conflict、MUFU、动态分支 |
| Sleeping                           | warp 中线程处于 blocked/yield/sleep       | `__nanosleep()`、等待策略、线程分组          |
| TEX Throttle                       | texture 指令队列腾出空间                  | texture/surface 指令过多、访问宽度过小       |
| Wait                               | 固定延迟执行指令的依赖完成                | 指令依赖链、ILP 不足、活跃 warp 不足         |
| Selected                           | 已被选择并成功发射指令                    | 正常状态，不是 stall                         |

## 容易混淆的几组指标

| 容易混淆的指标                     | 核心区别 |
| ---------------------------------- | -------- |
| LG Throttle vs Long Scoreboard     | LG 是“global/local 指令队列满”；Long 是“已发出的 L1TEX 操作数据还没回来”。 |
| MIO Throttle vs Short Scoreboard   | MIO 是“MIO 指令队列满”；Short 是“已发出的 MIO 操作结果还不能使用”。 |
| Barrier vs Membar                  | Barrier 等其他 warp 到达 CTA 同步点；Membar 等 memory fence 的可见性/顺序要求完成。 |
| Math Pipe Throttle vs Wait         | Math Pipe 是执行流水线吞吐不够；Wait 是固定延迟指令的数据依赖还未解除。 |
| Not Selected vs 其他 stall         | Not Selected 的 warp 已经 eligible；其他 stall 通常表示 warp 还不能发射。 |

## 1. 同步与控制流

### Barrier

**含义：** warp 已到达 CTA barrier，但同一 block 中还有 warp 没到，例如等待 `__syncthreads()`。

**常见代码原因：**

- 同步点之前存在不同长度的循环或不同工作量。
- 部分 warp 需要处理边界，其他 warp 不需要。
- block 很大，warp 到达同步点的时间差扩大。
- tile 循环中放置了过多同步。

**联动检查：** `Branch Efficiency`、每个 block 的线程数、Source/SASS 中 barrier 位置、`Eligible Warps Per Scheduler`。

**解决思路：**

- 删除算法上不需要的 `__syncthreads()`，但不能破坏正确性。
- 让同一 block 内各 warp 的工作量更均匀。
- 优化 barrier 之前最慢的代码，而不是只移动 barrier。
- 对可行的通信范围使用更小粒度的同步，例如 warp-level primitive。
- 对大 block 实验更小的 block，检查 Duration 是否稳定下降。

### Branch Resolving

**含义：** warp 正在等待分支目标计算完成以及程序计数器更新。

**常见代码原因：** 频繁跳转、嵌套条件、间接分支、复杂循环控制。它与 warp divergence 有关，但二者不是同一个指标。

**联动检查：** `Branch Efficiency`、`Avg. Active Threads Per Warp`、Source 中的分支热点、`No Instructions`。

**解决思路：** 合并重复条件、把 warp 一致的条件移到外层、减少热循环中的分支，并用实际编译后的 SASS 确认编译器是否已经使用 predication。

### No Instructions

**含义：** warp 没有取到下一条指令，可能正在等待 instruction fetch 或 instruction cache miss。

**常见代码原因：** kernel 很短且不足一个完整 wave、grid 太小、过度展开导致代码体积过大、执行路径跨越大量 SASS。

**联动检查：** `Waves Per SM`、Grid Size、Duration、instruction cache 指标、`Branch Resolving`。

**解决思路：** 增加可并行工作规模；减少没有收益的展开和代码复制；简化大范围跳转。短 kernel 中该项高可能只是采样现象，必须先看绝对执行时间。

### Sleeping

**含义：** warp 中所有线程都处于 blocked、yield 或 sleep 状态。

**常见代码原因：** `__nanosleep()`、自旋等待中的退避逻辑、不同线程采用不同等待策略。

**解决思路：** 减少 sleep 次数和时长；让同一 warp 的线程同时进入等待；能用同步原语或分阶段 kernel 解决时，避免长时间设备端轮询。

## 2. Memory 指令队列压力

### LG Throttle

**含义：** L1 中用于 local/global memory 操作的指令队列已满，warp 暂时无法继续发射 LG 指令。

**常见代码原因：**

- 热循环中 global load/store 指令过多。
- 相同数据被重复从 global memory 读取。
- 动态索引局部数组或寄存器压力导致 local-memory spill。
- 使用大量窄访问，本可合并为更少的宽访问。

**联动检查：** global/local load/store 指令数、L1/L2 throughput、local-memory bytes、寄存器 spill、DRAM throughput。

**解决思路：**

- 使用寄存器或 shared memory 复用数据。
- 消除重复 load/store，适用时使用对齐的向量化访问。
- 检查 `ptxas -v` 和 Source/SASS 中的 local load/store，处理 spill。
- 在 memory 与计算指令之间增加独立工作，但不要仅为降低 stall 而增加无效计算。

`LG Throttle` 高而 `DRAM Throughput` 低并不矛盾：瓶颈可以是指令队列，而不是显存带宽。

### MIO Throttle

**含义：** MIO 指令队列已满。MIO 路径包括 shared-memory 指令、部分特殊数学指令和动态分支。

**常见代码原因：** shared-memory 访问次数过多、每次访问宽度太小、register tiling 不足、特殊数学或动态跳转密集。

**联动检查：** Shared Memory 表、shared load/store 指令数、MIO pipeline utilization、`Short Scoreboard`、bank conflict。

**解决思路：**

- 用寄存器保存高频复用值，减少 shared-memory 指令数。
- 适用时合并为更少、更宽的 shared load/store。
- 使用 register tiling，让每次 shared load 支撑更多计算。
- 分别检查 shared、MUFU 和动态分支，不能看到 MIO 就直接归因于 shared memory。

bank conflict 会增加 shared-memory 实际 wavefront，但 `MIO Throttle` 高不等于一定存在 bank conflict。

### TEX Throttle

**含义：** texture operation 的 L1 指令队列已满。

**常见代码原因：** texture fetch、surface load/store 或相关操作发射过于频繁。

**解决思路：** 删除重复 texture 操作；适用时合并窄访问；在 texture 与独立计算之间交错；如果不需要 texture 的寻址、过滤或 cache 行为，实验改用合并良好的 global memory 访问。

## 3. 数据依赖等待

### Long Scoreboard

**含义：** warp 下一条指令依赖某个 L1TEX 操作的结果，而 global/local/surface/texture 数据尚未准备好。

**常见代码原因：** 非合并访问、cache miss、访问局部性差、加载后立即使用、可并行 warp 或独立指令不足。

**联动检查：** DRAM/L2/L1 throughput、cache hit rate、global load efficiency、Memory Workload Analysis、occupancy 和 eligible warps。

**解决思路：**

- 让 warp 访问连续且对齐的地址，减少不必要的 sectors。
- 增加数据局部性，把重复使用的数据放入 shared memory 或寄存器。
- 通过预取、循环重排或双缓冲增加 memory 与 compute 的重叠。
- 在资源允许时增加 ILP 或 active warps 来隐藏延迟。

### Short Scoreboard

**含义：** warp 正在等待非 L1TEX 的 MIO 操作结果，最常见的是 shared-memory 数据依赖，也可能来自 MUFU 特殊数学或动态分支。

**联动检查：** shared bank conflicts、shared wavefronts/requests、MIO utilization、Source/SASS、`MIO Throttle`。

**解决思路：** 消除 bank conflict；减少 shared-memory 往返；把高频值保存在寄存器；重排指令以增加独立计算；必要时使用双缓冲。

### IMC Miss

**含义：** warp 等待 immediate constant cache miss。SASS 中常见形式是 `c[bank][offset]`。

**常见代码原因：** warp 内线程读取很多不同的 constant 地址，导致访问被序列化，或者常量工作集不能有效命中 cache。

**解决思路：** 让同一 warp 尽量读取相同或少量不同的 constant 地址；把高频常量提前读入寄存器；对于线程间地址高度分散的数据，实验使用更合适的 global/shared 存储方式。

### Wait

**含义：** warp 等待固定延迟执行指令的依赖完成，通常反映较长的串行依赖链，而不是 memory queue 满。

**联动检查：** Source/SASS dependency、active/eligible warps、occupancy、循环展开程度、指令类型。

**解决思路：** 增加相互独立的累加器或其他 ILP；合理展开循环；增加可隐藏延迟的 active warps；适用时采用低延迟指令或 fast-math。改变数学语义前必须验证精度。

## 4. 执行流水线与调度

### Math Pipe Throttle

**含义：** warp 的下一条数学指令已经准备好，但对应执行流水线没有空位。

**常见代码原因：** 大量 warp 同时依赖同一种数学 pipeline，例如 FP32、Tensor 或特殊函数路径，指令组合过于集中。

**联动检查：** `Compute Workload Analysis` 中各 pipeline utilization、实际 FLOP/s、Roofline、Duration。

**解决思路：** 平衡可替代的指令组合；提高每条昂贵指令产生的有效工作；适用时使用 Tensor Core 或更合适的精度；通过其他 warp/独立指令隐藏 pipeline latency。pipeline 已达到吞吐上限时，单纯增加 occupancy 通常不会提高吞吐。

### Dispatch Stall

**含义：** 指令已经可以发射，但 dispatcher 因冲突或硬件事件暂缓发射。

**解决思路：** 该名称本身不提供足够的源码原因。先在 Source/SASS 中定位高采样指令，再结合 pipeline utilization、指令组合和相邻 stall 判断。不要仅针对 `Dispatch Stall` 做无依据改写。

### Not Selected

**含义：** warp 已经 eligible，但调度器本周期选择了另一个 warp。

这通常不是坏事。较高的 `Not Selected` 往往说明有足够多 eligible warps，调度器能够用其他 warp 覆盖延迟。

**处理方式：** 不要把降低它作为优化目标。如果 Duration 已低且 issue slot 使用充分，可以不处理。资源竞争严重时，可实验降低 active warps 来换取寄存器复用或 cache 局部性，但最终只看 Duration 和工作量是否改善。

### Selected

**含义：** warp 被调度器选中并成功发射了指令。

它出现在 Warp State Statistics 中是为了构成完整的 warp 状态分布，但它不是 stall reason，也不需要“解决”。

## 5. Memory 顺序与 kernel 收尾

### Membar

**含义：** warp 正在等待 memory barrier/fence 要求满足，例如等待之前的 memory 操作达到所需可见性或顺序。

**常见代码原因：** 不必要或范围过大的 fence、频繁的原子同步、fence 前积累了大量尚未完成的 memory 操作。

**解决思路：** 删除语义上冗余的 fence；使用满足正确性的最小作用域；减少 fence 前的低效 memory 操作。不能为了性能直接移除保证正确性的 memory ordering。

### Drain

**含义：** warp 已执行 `EXIT`，但仍需等待 outstanding memory 操作完成，之后才能释放 warp 资源。

**常见代码原因：** kernel 尾部集中执行大量 store、写地址未合并、线程结束时间差异较大。

**解决思路：** 合并并对齐尾部 store；减少不必要写回；优化 reduction 和结果写出；让各 warp 尾部工作量更均衡。

### Misc

**含义：** NCU 没有归入其他类别的硬件等待。

**处理方式：** 先排除更明确的 stall；用 Source/SASS 找高采样行；检查 NCU 其他 section、工具版本和架构说明。`Misc` 不是可直接对应某一种代码修改的结论。

## 架构相关补充：Warpgroup Arrive

新版 NCU 还可能列出 `Warpgroup Arrive`，表示等待 `WARPGROUP.ARRIVE` 或 `WARPGROUP.WAIT` 一类指令。它主要与支持 warpgroup 异步协作的新架构有关，RTX 4090 通常不会使用这类指令。

如果当前服务器查询不到该指标，属于正常情况，不应把它加入 4090 的固定记录表。

## 实际分析步骤

拿到 `.ncu-rep` 后，按照下面的顺序判断：

```text
1. 先确认 Duration 是否稳定，结果是否正确。
2. 查看 Scheduler Statistics：是否存在大量未使用的 issue slots。
3. 查看 Eligible Warps Per Scheduler：是否经常没有可发射 warp。
4. 在 Warp State Statistics 中记录最大的 2~3 项，不只记录第一项。
5. 同时记录 cycles per issued instruction 和占比。
6. 在 Source/SASS 中定位对应 stall 最高的代码行和指令。
7. 用 Memory、Compute、Occupancy 等 section 验证原因。
8. 每次只改变一个因素，比较 Duration 和对应 stall 的绝对周期。
```

可以从完整报告中快速搜索 stall 指标：

```bash
ncu --import report.ncu-rep --page raw --csv \
  | grep -Ei 'warp_issue_stalled|Warp Cycles Per Issued Instruction'
```

如果只看 UI：

```text
Details
  -> Scheduler Statistics
  -> Warp State Statistics
  -> Source（查看 CUDA/SASS 热点）
```

## 判断解决方案是否有效

不能使用“Top Stall 占比下降”作为唯一证据。一次有效优化至少应满足：

| 证据层级                           | 应看到的结果 |
| ---------------------------------- | ------------ |
| 正确性                             | 输出结果与 baseline 一致 |
| 性能                               | 多轮 benchmark 的中位 Duration 稳定下降，且超过波动范围 |
| 硬件因果                           | 目标 stall 的绝对 cycles 或相关指令/请求减少 |
| 无严重副作用                       | 没有产生更大的新 stall、spill、occupancy 或流量问题 |

stall 占比是相对值。某项 stall 的绝对周期下降后，因为其他部分下降得更多，它的占比仍可能升高。因此比较两个 kernel 时，应优先看：

```text
Duration
+ stall cycles per issued instruction
+ 相关指令、requests、bytes、wavefronts 的绝对工作量
```

## 当前 GEMM 示例

| Kernel                             | Top Stall                            | 代码与硬件解释 |
| ---------------------------------- | ------------------------------------ | -------------- |
| `gemm_naive`                       | `LG Throttle: 27.0 cycles, 65.6%`   | K 循环反复发射 global load，LG 指令队列压力大；DRAM 很低说明不是显存带宽跑满。 |
| `gemm_tiled`                       | `MIO Throttle: 20.4 cycles, 51.5%`  | 数据复用转到 shared memory，LG 压力下降，但 shared-memory 指令使 MIO 队列成为主要压力。 |

下一步实验应比较 `gemm_tiled` 与 register-tiled 版本：如果 shared load/store 指令数和 MIO stall 同时下降，而且 Duration 稳定下降，才能证明 register tiling 缓解了 MIO 压力。

## 官方参考

- [NVIDIA Nsight Compute Profiling Guide: Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)
- [NVIDIA Nsight Compute Profiling Guide: Warp State Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Nsight Compute UI Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
