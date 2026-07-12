# NCU 从 GEMM 演进到陌生算子的分析工作流

当前学习阶段不需要立刻把算子当成完全黑盒。工程里已经准备了多个 GEMM 版本，它们提供了一组可控实验：数学目标基本相同，但数据复用、shared memory、寄存器、访问宽度和计算单元逐步发生变化。

因此采用两阶段路线：

```text
阶段一：白盒学习
阅读一组 GEMM 对照版本的代码差异
  -> 写下对八个指标的变化预期
  -> 查看 NCU 验证预期
  -> 建立“代码 -> 指令/流量 -> 指标”的映射

阶段二：黑盒迁移
暂时不看陌生 kernel 的实现
  -> 根据指标提出候选假设
  -> 用补充指标缩小范围
  -> 再打开 Source/SASS 验证
  -> 用单变量实验确认因果
```

这样既能利用已知代码快速理解指标，又不会长期停留在“带着答案找证据”的状态。

## 当前学习路线

| 学习阶段        | 主要材料                             | 达成目标                                    |
| --------------- | ------------------------------------ | ------------------------------------------- |
| 1. 建立概念     | `gemm_naive` 与 `gemm_tiled`         | 能准确读取八个指标，并解释 LG 到 MIO 的迁移 |
| 2. 建立因果     | padding、register tile、float4、WMMA | 能先预测指标，并区分单变量与混杂变量        |
| 3. 去掉上帝视角 | 隐藏 GEMM 版本名或分析其他类型算子   | 能只凭报告提出多个候选，而不是直接猜实现    |

阶段一允许看源码，但每次必须先写预测再看报告。阶段二则反过来，先看报告形成假设，再打开源码。

## 工程中的 GEMM 实验梯子

源码位置：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)。固定使用相同 shape 比较时，版本之间的差异更容易与 NCU 指标对应。

| GEMM 版本                | 相比对照版本的主要变化                               | 最适合学习的指标                                   |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------------- |
| `gemm_naive`             | 每线程计算一个输出；K 循环直接读取 A/B global memory | Duration、Memory、L2、DRAM、LG Throttle            |
| `gemm_tiled`             | A/B tile 搬入 shared memory，在 block 内复用         | Duration、Memory 层级、Top Stall 切换              |
| `gemm_tiled_padding`     | shared 数组第二维从 16 改为 17                       | Top Stall、bank conflict；八指标之外需看 shared 表 |
| `gemm_regtile2x2`        | 每线程计算 2x2 输出，增加寄存器累加和数据复用        | Registers、Occupancy、MIO、Duration                |
| `gemm_regtile4x4`        | 每线程计算 4x4 输出，进一步增加 ILP 和寄存器         | Registers 与 Occupancy 的交换、spill 风险          |
| `gemm_vectorized_float4` | 每线程计算四列，B 使用对齐 `float4` 读取             | LG Throttle、Registers、指令数量、Duration         |
| `gemm_wmma_fp16`         | FP16 输入并使用 WMMA/Tensor Core                     | Compute pipeline、Duration、Roofline               |

这七个版本不能简单理解为从上到下必然越来越快：

- `gemm_tiled_padding` 是一个假设验证版本。经典的 `+1 padding` 不一定适合当前 `16x16` 访问方式，必须以 bank-conflict 指标和 Duration 为准。
- `gemm_vectorized_float4` 没有沿用 tiled/shared-memory 路径，它是从直接 global-memory 计算分出的另一条实验支线。
- `gemm_wmma_fp16` 同时改变了数据类型和计算单元，不能作为严格的单变量实验与 FP32 版本直接归因比较。
- 优化可能让某个指标变差但 Duration 变好，例如 register tiling 增加寄存器并降低 occupancy，却减少了 shared-memory 指令。

## 八个指标分别用哪组 GEMM 学

| 关键指标                | 首选对照实验                       | 本轮要回答的问题                                        |
| ----------------------- | ---------------------------------- | ------------------------------------------------------- |
| Duration                | 六组指定对照                       | 代码改动是否带来稳定且超过波动的加速                    |
| Compute (SM) Throughput | regtile 版本、WMMA 与 FP32 版本    | 高值来自哪条 pipeline，是否等于有效计算更快             |
| Memory Throughput       | naive -> tiled                     | 高值由 global、L1/TEX 还是 shared/MIO 驱动              |
| DRAM Throughput         | naive -> tiled，并补充不同矩阵规模 | global load 多是否真的等于 DRAM 带宽高                  |
| L2 Throughput           | naive -> tiled                     | shared-memory 复用是否减少 L2 路径压力                  |
| Achieved Occupancy      | tiled -> regtile2x2 -> regtile4x4  | active warp 减少是否真的造成 eligible warp 不足         |
| Registers / Thread      | tiled -> regtile2x2 -> regtile4x4  | 寄存器增加换来了什么复用，是否产生 occupancy/spill 代价 |
| Top Stall Reason        | naive -> tiled -> regtile          | 主要等待如何随 global、shared、register 路径迁移        |

不要要求每次代码改动都让八个指标明显变化。一轮实验只重点理解 2~4 个指标，其余指标用于检查副作用。

## 推荐实验顺序

| 轮次 | 对照版本                                                         | 主要学习目标                                                  |
| ---- | ---------------------------------------------------------------- | ------------------------------------------------------------- |
| 1    | [naive vs tiled](gemm_naive_vs_tiled.md)                         | global 读取变为 shared 复用后，Memory/L2/Top Stall 如何变化   |
| 2    | [tiled vs tiled_padding](gemm_tiled_vs_tiled_padding.md)         | 如何验证 bank conflict，而不是看到 padding 就预设有效         |
| 3    | [tiled vs regtile2x2](gemm_tiled_vs_regtile2x2.md)               | shared 访问减少与寄存器增加之间的交换                         |
| 4    | [regtile2x2 vs regtile4x4](gemm_regtile2x2_vs_regtile4x4.md)     | ILP、Registers、Occupancy 和 spill 的平衡                     |
| 5    | [naive vs vectorized_float4](gemm_naive_vs_vectorized_float4.md) | 宽访问和每线程多输出如何影响 LG 指令压力                      |
| 6    | [最佳 FP32 版本 vs wmma_fp16](gemm_best_fp32_vs_wmma_fp16.md)    | 识别 Tensor pipeline 和 Roofline；同时记录 dtype 这一混杂变量 |

### 第 1 轮：naive vs tiled

当前已经完成第一轮基础数据采集。重点不是记住 `3.82 ms -> 2.93 ms`，而是掌握下面的因果链：

```text
global load 指令减少，数据转到 shared memory 复用
  -> L2 压力下降但 Memory 汇总值仍可能很高
  -> DRAM 始终很低，排除 DRAM 带宽饱和
  -> Top Stall 从 LG Throttle 切换到 MIO Throttle
  -> Duration 稳定下降，支持改动有效
```

### 第 2 轮：[tiled vs tiled_padding](gemm_tiled_vs_tiled_padding.md)

先写下两种相反假设：

```text
H1：原 tiled 存在 bank conflict，padding 后冲突和 Duration 下降。
H2：原 tiled 已基本无冲突，padding 没有收益，甚至产生新的访问冲突。
```

八指标负责观察 Duration、Memory 和 Top Stall，但最终必须补充 Shared Memory 表中的 bank conflicts、requests 和 wavefronts。该实验用于学习“八指标初筛后为什么还要补指标”。

### 第 3~4 轮：register tiling

- [第 3 轮：tiled vs regtile2x2](gemm_tiled_vs_regtile2x2.md)
- [第 4 轮：regtile2x2 vs regtile4x4](gemm_regtile2x2_vs_regtile4x4.md)

代码从每线程一个输出，扩展到每线程 2x2、4x4 输出。实验前预期：

```text
可能改善：shared 指令/每次 FMA、MIO Throttle、Duration
可能变差：Registers / Thread、Achieved Occupancy、spill 风险
```

如果 Registers 增加、Occupancy 下降，但 Eligible Warps 仍足够且 Duration 降低，这是一种成功的资源交换，不应因为 occupancy 下降就否定优化。

### 第 5 轮：[naive vs vectorized_float4](gemm_naive_vs_vectorized_float4.md)

这个版本应与 naive 对照，不与 regtile4x4 视为单线演进。重点验证：更少、更宽的 B load 是否减少 LG 指令压力，以及四个累加器是否增加 Registers / Thread。

### 第 6 轮：[最佳 FP32 版本 vs wmma_fp16](gemm_best_fp32_vs_wmma_fp16.md)

这轮用于认识 Tensor Core、Compute Workload Analysis 和 Roofline，不用于做严格的 FP32 单变量归因。报告中必须同时记录输入 dtype、Tensor pipeline、实际 TFLOP/s 和 Duration。

## 每轮固定操作

每个 GEMM 版本都使用同一 shape、同一 GPU 环境和同一套步骤：

```text
1. 阅读两个版本之间真正改变的代码。
2. 只写 1~3 条指标预测，预测可以错误。
3. 先用 CUDA Event benchmark 确认时间和波动。
4. 再采集 NCU full report，并提取八个指标。
5. 检查补充 section，验证 Top Stall 对应的硬件路径。
6. 写出“代码 -> 工作量/指令 -> 指标 -> Duration”因果链。
7. 如果证据不支持预测，保留反例并修改理解。
```

在 4090 服务器上可以依次采集：

```bash
# 先用 CUDA Event 查看各 shape 的稳定时间
python benchmark/bench_gemm_shapes.py

# 再为固定的 2048x2048x2048 输入采集 NCU 报告
ITERS=1 bash scripts/profile_ncu_full.sh gemm_naive
ITERS=1 bash scripts/profile_ncu_full.sh gemm_tiled
ITERS=1 bash scripts/profile_ncu_full.sh gemm_tiled_padding
ITERS=1 bash scripts/profile_ncu_full.sh gemm_regtile2x2
ITERS=1 bash scripts/profile_ncu_full.sh gemm_regtile4x4
ITERS=1 bash scripts/profile_ncu_full.sh gemm_vectorized_float4
ITERS=1 bash scripts/profile_ncu_full.sh gemm_wmma_fp16
```

`profile_entry.py` 会先执行 5 次预热。提取报告时通常选择第 6 次 invocation，并确认选中的 kernel 名称与版本一致。

例如提取 `regtile2x2` 的八个指标：

```bash
bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_regtile2x2_full.ncu-rep \
  gemm_regtile2x2_kernel \
  6
```

每轮使用下面的最小记录格式：

```text
【代码唯一主要变化】

【运行前预测】
预计 Duration：
预计重点指标：
预计 Top Stall：

【实际八指标】
Duration / Compute / Memory / DRAM / L2：
Occupancy / Registers / Top Stall：

【补充证据】
Scheduler / Compute pipeline / Memory table / Source-SASS：

【结论】
预测得到支持还是被推翻：
能够确认的因果：
仍不能确认的问题：
```

## 从 GEMM 白盒过渡到黑盒

完成每组 GEMM 对照后，不要直接进入下一个版本。先做一次“隐藏答案复盘”：

1. 暂时隐藏 kernel 名称和源码，只保留 NCU 数据。
2. 写出至少两个候选解释，并指出还缺什么证据。
3. 再揭示版本名称，检查推理是否依赖了源码先验。
4. 最后打开 Source/SASS，将指标定位到实际指令。

当你能够根据报告说出“下一步应该查哪个 section 或做哪个实验”，而不是立即猜中源码实现时，就具备了迁移到 softmax、normalization 和 attention 的基础。

## 通用分析方法：面对陌生算子

进入第二阶段后，分析的核心不是“根据指标猜答案”，而是：

```text
观察事实
  -> 提出多个候选假设
  -> 用下一组指标排除假设
  -> 回到 Source/SASS 定位代码
  -> 设计单变量实验
  -> 用正确性和 Duration 验证因果
```

## 一、先改变结论标准

实际分析应区分四个证据等级：

| 等级     | 当前能说什么                             | 示例                                       |
| -------- | ---------------------------------------- | ------------------------------------------ |
| L0：观察 | 只描述报告中的客观数值                   | DRAM Throughput 是 1.3%                    |
| L1：候选 | 根据多个指标提出可能方向                 | 可能不是 DRAM 带宽受限                     |
| L2：定位 | 补充指标和 Source/SASS 指向具体硬件路径  | shared 指令多且 MIO 队列繁忙               |
| L3：因果 | 单变量改动使目标指标和 Duration 同时改善 | 减少 shared load 后 MIO stall 和时间都下降 |

在到达 L2 之前，尽量使用“候选”“怀疑”“需要检查”，不要直接写“该 kernel 是某某 bound”。真正可信的优化结论应达到 L3。

## 二、确认分析对象与测量可信

即使把 kernel 当作黑盒，也必须知道它的调用契约。分析前记录：

```text
Kernel 名称：
GPU / CUDA / NCU 版本：
输入 shape：
数据类型：
内存 layout / stride：
block / grid：
动态 shared memory：
是否包含预热：
输出是否通过正确性检查：
多轮 benchmark 中位时间和波动范围：
```

必须分开两个任务：

- **Benchmark** 判断优化是否真的变快，使用 CUDA Event、预热和多轮统计。
- **NCU Profile** 解释某一次稳定 kernel 执行时发生了什么。

NCU 会多 pass 重放 kernel，不能把一次 NCU Duration 当作完整的稳定性 benchmark。

## 三、第一步只做高层初筛

先记录已有的八个指标，但不立即下最终结论：

| 指标                    | 第一眼回答的问题             | 不能单独证明什么                     |
| ----------------------- | ---------------------------- | ------------------------------------ |
| Duration                | kernel 完成工作需要多久      | 不能解释为什么快或慢                 |
| Compute (SM) Throughput | 是否有某条 SM 路径很繁忙     | 不能证明 FP32/Tensor 已达到峰值      |
| Memory Throughput       | memory 子系统是否有繁忙路径  | 不能证明 DRAM 已跑满                 |
| DRAM Throughput         | 显存数据通路是否繁忙         | 低值不能排除 cache、队列或延迟问题   |
| L2 Throughput           | L2 数据通路是否繁忙          | 不能区分有效流量和低效请求           |
| Achieved Occupancy      | 实际 active warps 是否充足   | 高值不能证明 eligible warps 充足     |
| Registers / Thread      | 寄存器分配是否可能限制并行度 | 不能单独证明存在 spill               |
| Top Stall Reason        | warp 最常处于哪种等待状态    | 不能证明它是唯一瓶颈或可获得同等加速 |

这一步的产出不是 `Compute-Bound` 或 `Memory-Bound`，而是下面几类候选：

```text
A. 工作规模不足 / launch 与尾部效应
B. 某条计算 pipeline 吞吐受限
C. DRAM、L2、L1/TEX 或 shared memory 吞吐受限
D. memory 或执行指令的数据依赖延迟
E. 同步、分支或负载不均衡
F. 寄存器、shared memory 或 occupancy 资源限制
G. 多种因素混合，需要实验分离
```

## 四、先经过 Scheduler 分流

八个指标不足以完成黑盒判断。下一步优先打开 `Scheduler Statistics`，因为它能区分“硬件忙”与“warp 根本发不出来”。

重点查看：

```text
Active Warps Per Scheduler
Eligible Warps Per Scheduler
Issued Warps Per Scheduler
No Eligible / Skipped Issue Slots
Warp Cycles Per Issued Instruction
```

| Scheduler 现象               | 初步方向                             | 下一步                             |
| ---------------------------- | ------------------------------------ | ---------------------------------- |
| Active 少，Eligible 也少     | grid 太小、occupancy 或资源限制      | 看 Waves Per SM、Launch、Occupancy |
| Active 多，Eligible 很少     | warp 大量等待依赖、同步或数据返回    | 看 Warp State Statistics           |
| Eligible 多，Issued 接近上限 | scheduler 有活可发，更像吞吐路径繁忙 | 看 Compute/Memory 各子流水线       |
| Not Selected 高              | 通常说明 eligible warp 充足          | 不要把它直接当成坏事               |
| issue slots 经常空闲         | 延迟隐藏失败或并行工作不足           | 结合 Top 2~3 stall 继续定位        |

`Achieved Occupancy` 高只说明 active warp 多。只有 `Eligible Warps` 也足够，才能说明这些 warp 真正具备发射条件。

## 五、如果像是 memory 问题，定位到具体层级

不要使用笼统的 `Memory-Bound`。按下面的层级继续拆分：

```text
global/local/shared memory 指令
  -> L1/TEX 或 shared 路径
  -> L2
  -> DRAM
```

### 5.1 先看哪一层接近峰值

| 观察组合               | 候选判断                     | 需要补充的证据                        |
| ---------------------- | ---------------------------- | ------------------------------------- |
| DRAM 高                | DRAM bandwidth-bound 候选    | Roofline、DRAM bytes、访问效率        |
| L2 高、DRAM 低         | L2/cache 数据通路候选        | L2 sectors、hit rate、请求效率        |
| L1/TEX 高、L2/DRAM 低  | 片上 L1/TEX 或指令路径候选   | L1 requests、LG/TEX stall             |
| Shared/MIO 高、DRAM 低 | shared-memory/MIO 候选       | shared 指令、wavefront、bank conflict |
| 各层 throughput 都低   | 更像延迟、同步或工作规模不足 | Scheduler、Scoreboard、Waves Per SM   |

### 5.2 再区分“队列满”和“数据没回来”

| Stall 组合       | 它真正表示什么                    | 优先实验                                 |
| ---------------- | --------------------------------- | ---------------------------------------- |
| LG Throttle      | global/local memory 指令队列满    | 减少指令数、数据复用、宽访问、检查 spill |
| Long Scoreboard  | L1TEX 操作的数据依赖尚未完成      | 合并访问、局部性、预取、ILP              |
| MIO Throttle     | shared/MIO 等指令队列满           | 减少 shared 指令、register tiling        |
| Short Scoreboard | shared/MIO 操作的数据依赖尚未完成 | bank conflict、寄存器复用、指令重排      |

这四项不能互相替代：Throttle 更偏**吞吐和队列容量**，Scoreboard 更偏**已发出操作的结果延迟**。

### 5.3 检查“做了多少工作”，不只看百分比

Throughput 是单位时间利用率。优化后 Duration 缩短，即使总流量减少，Throughput 百分比也可能不降。

比较两个版本时还要记录：

```text
DRAM / L2 bytes
sectors 与 requests
global/local/shared 指令数
shared wavefronts / requests
cache hit rate
local-memory load/store 与 spill
```

## 六、如果像是 compute 问题，定位到具体 pipeline

`Compute (SM) Throughput` 高不能直接证明浮点计算受限。打开 `Compute Workload Analysis`，确认是哪条 pipeline 高：

```text
FP32 / FP64
Tensor / HMMA
INT
MUFU / special math
LSU
其他架构相关 pipeline
```

形成 Compute-Bound 结论至少需要三类证据：

1. 对应数学 pipeline 接近持续峰值，而不是只有 SM 汇总值高。
2. Roofline 中算术强度位于 compute 区域，并接近对应计算 roof。
3. Scheduler 仍有足够 eligible warps，主要问题不是等待数据或同步。

可验证的实验包括：减少数学操作、改用 Tensor Core、改变精度、减少特殊函数或增加每条指令产生的有效计算。只有 Duration 按预期下降，才能确认计算路径是主要限制。

## 七、如果 Compute 和 Memory 都不高

这时不能直接叫 `Latency-Bound`，应按顺序排查：

| 排查项       | 典型证据                                 | 可能的下一步                               |
| ------------ | ---------------------------------------- | ------------------------------------------ |
| 工作规模不足 | Waves Per SM 低、grid 小、kernel 很短    | batch/fusion、增加并行工作、减少 launch    |
| 数据依赖延迟 | Eligible 少、Long/Short Scoreboard 高    | 预取、ILP、访问优化、增加可隐藏延迟的 warp |
| 固定指令延迟 | Wait 高、串行 dependency chain           | 多累加器、合理展开、低延迟指令             |
| 同步等待     | Barrier/Membar 高                        | 均衡同步前工作、减少不必要同步             |
| 控制流问题   | Branch Resolving/No Instructions 高      | 简化分支、检查 divergence 和代码体积       |
| 资源限制     | theoretical occupancy 低、有明确 limiter | block size、寄存器、shared memory 实验     |

## 八、Occupancy 和 Registers 只用于解释限制

不要把 occupancy 最大化当作目标。正确的判断链是：

```text
Registers / Thread 或 Shared Memory / Block 增加
  -> Theoretical Occupancy 是否下降
  -> Achieved Active Warps 是否下降
  -> Eligible Warps 是否因此不足
  -> Scoreboard/Wait 是否增加
  -> Duration 是否变差
```

如果 register tiling 让 occupancy 从 100% 降到 60%，但减少了大量 memory 指令，并使 Duration 降低，那么这是有效交换。

如果 Registers / Thread 下降但出现 local-memory load/store，反而要怀疑编译器 spill 或代码结构改变。需要结合 `ptxas -v` 和 Source/SASS，而不是只看寄存器数字。

## 九、最后才回到 Source/SASS

黑盒分析不等于永远不看源码，而是**先由指标缩小范围，再带着可验证问题看源码**。

错误方式：

```text
我看到源码用了 shared memory
-> 所以 MIO 一定来自 shared memory
```

更可靠的方式：

```text
MIO Throttle 高
-> Shared Memory 表显示 shared 指令和 wavefront 很多
-> Source/SASS 热点落在 LDS/STS 指令
-> 减少 shared load 后 MIO cycles 和 Duration 同时下降
```

在 Source 页面重点做三件事：

1. 按 stall sampling 或执行周期定位最热代码行。
2. 切换 SASS，确认实际是 global、local、shared、数学还是同步指令。
3. 检查源码行与多条 SASS 的对应关系，避免把编译器生成行为误判为源码表面行为。

## 十、用单变量实验完成因果验证

每次实验只改变一个主要因素，并提前写下预期：

| 当前假设                      | 单变量实验                           | 预期同时变化的证据                                   |
| ----------------------------- | ------------------------------------ | ---------------------------------------------------- |
| DRAM 带宽限制                 | 提高数据复用或算术强度               | DRAM bytes/压力下降，Duration 下降                   |
| global 指令队列限制           | 寄存器/shared 复用或对齐宽访问       | LG 指令数、LG Throttle、Duration 下降                |
| global 数据延迟               | 合并访问、预取、增加独立累加器       | Long Scoreboard 和 Duration 下降                     |
| shared 指令吞吐限制           | register tiling，减少 shared load    | shared 指令、MIO Throttle、Duration 下降             |
| shared bank conflict          | 只改变 shared layout/padding         | excessive wavefront、Short Scoreboard、Duration 下降 |
| 数学 pipeline 限制            | 改变指令类型、精度或使用 Tensor Core | 目标 pipeline 压力和 Duration 变化                   |
| Barrier 等待                  | 均衡同步前工作或减少一次同步         | Barrier cycles 和 Duration 下降                      |
| occupancy/latency hiding 不足 | 只改变 block size 或资源使用         | Eligible Warps、Scoreboard、Duration 改善            |
| grid 太小                     | 增加独立工作或与相邻操作 fusion      | Waves Per SM/总耗时改善                              |

如果目标 stall 降低但 Duration 没有下降，可能是：

- 原 stall 已被其他 warp 隐藏，本来就不是关键限制。
- 新的瓶颈接替了原瓶颈。
- 改动减少 stall 的同时增加了指令、寄存器或同步成本。
- 差异仍在测试波动范围内。

## 十一、完整决策顺序

```text
结果正确、benchmark 稳定吗？
  否 -> 先修正测试
  是
  |
目标 kernel 值得优化吗？Duration/调用占比足够大吗？
  否 -> 优先看 fusion、调用次数或其他 kernel
  是
  |
Waves Per SM 和 Active Warps 足够吗？
  否 -> 工作规模、launch、block 配置或资源限制
  是
  |
Scheduler 是否经常没有 Eligible Warp？
  是 -> Scoreboard / Wait / Barrier / Branch 路径
  否 -> 吞吐路径
         |
         +-> 具体 compute pipeline 高 -> Compute 候选
         +-> DRAM 高                 -> DRAM bandwidth 候选
         +-> L2/L1 高                -> cache/on-chip memory 候选
         +-> MIO/shared 高           -> shared/MIO 候选
  |
Source/SASS 能否定位到对应指令？
  否 -> 证据不足，继续收集
  是
  |
单变量实验是否让目标指标和 Duration 同时改善？
  否 -> 推翻或降低该假设的优先级
  是 -> 得到因果结论，继续寻找新的最大限制
```

## 十二、用现有 GEMM 数据做隐藏答案复盘

完成 naive 与 tiled 的白盒学习后，暂时隐藏两个 kernel 的名称和实现，只看下面的报告。这一步用于检查自己能否从“已知答案解释指标”过渡到“根据指标选择下一步证据”。

### Kernel A

```text
Duration              3.82 ms
Compute Throughput    98.35%
Memory Throughput     98.35%
DRAM Throughput        1.09%
L2 Throughput         37.83%
Achieved Occupancy    98.09%
Registers / Thread    40
Top Stall             LG Throttle 27.0 cycles, 65.6%
```

不看源码时可以得到：

1. DRAM 只有 1.09%，暂不支持 DRAM 带宽受限。
2. occupancy 很高，但还不知道 eligible warps 是否充足。
3. LG Throttle 很高，支持 global/local memory 指令队列压力候选。
4. 下一步应查 global/local 指令数、local-memory spill、L1/L2 requests 和 Source/SASS 热点。

此时不能直接说“它在 K 循环中重复读取 A/B”，因为报告尚未提供这个源码事实。

### Kernel B

```text
Duration              2.93 ms
Compute Throughput    96.38%
Memory Throughput     96.38%
DRAM Throughput        1.32%
L2 Throughput         33.03%
Achieved Occupancy    98.20%
Registers / Thread    38
Top Stall             MIO Throttle 20.4 cycles, 51.5%
```

不看源码时可以得到：

1. 同样不支持 DRAM 带宽受限。
2. 主要等待从 LG 变成 MIO，说明繁忙硬件路径发生了迁移。
3. MIO 可能来自 shared memory、特殊数学或动态分支，当前不能只凭名称认定是 shared memory。
4. 下一步应查 Shared Memory 表、MIO pipeline、bank conflict 和 Source/SASS。

只有看到 shared 指令热点，并通过减少 shared 指令的实验改善 Duration，才完成从“候选”到“因果”的推理。

## 十三、每轮分析的记录模板

```text
【客观现象】
Duration：
Compute / Memory / DRAM / L2：
Occupancy / Registers：
Scheduler：
Top 3 Stall：

【候选假设】
H1：
H2：
H3：

【排除证据】
已经可以排除什么，为什么：

【下一组指标】
哪一个指标能区分 H1/H2/H3：

【Source/SASS 定位】
热点指令和代码行：

【单变量实验】
只改什么：
预期哪些指标变化：

【结果】
正确性：
Duration 中位数和波动：
硬件指标变化：

【结论等级】
L0 观察 / L1 候选 / L2 定位 / L3 因果
```

## 最终原则

```text
指标不是答案，而是用来选择下一步实验的证据。

不要问：哪个 stall 最大，所以应该改什么？
要问：现在有哪些候选原因，哪个补充指标或最小实验能最快区分它们？
```

## 相关文档

- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU Warp Stall 原因与解决思路](ncu_warp_stall_reasons.md)
- [NCU GEMM 入门：gemm_naive vs gemm_tiled](gemm_naive_vs_tiled.md)
- [第二组：gemm_tiled vs gemm_tiled_padding](gemm_tiled_vs_tiled_padding.md)
- [第三组：gemm_tiled vs gemm_regtile2x2](gemm_tiled_vs_regtile2x2.md)
- [第四组：gemm_regtile2x2 vs gemm_regtile4x4](gemm_regtile2x2_vs_regtile4x4.md)
- [第五组：gemm_naive vs gemm_vectorized_float4](gemm_naive_vs_vectorized_float4.md)
- [第六组：最佳 FP32 版本 vs gemm_wmma_fp16](gemm_best_fp32_vs_wmma_fp16.md)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
