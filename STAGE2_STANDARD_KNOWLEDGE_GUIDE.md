# 阶段 2 正常掌握版教材：从“会写 CUDA”到“能优化并证明”

> 版本：2.1，正常掌握、自包含教材、概念与关系双维版。
> 前置条件：已经通过阶段1，能够独立完成PyTorch CUDA Extension、Reduction、输入契约、测试和可信计时。  
> 主案例：GEMM优化链。  
> 迁移案例：Softmax。  
> 目标：每一次性能变化都形成“预测 → 实验 → 指标 → 结论 → 代价”的证据链，并把方法迁移到Norm、逐元素融合、Attention、量化和多实现工程。

## 教材使用说明

本教材不是优化技巧列表，也不是只列知识范围的验收提纲，而是阶段2的完整工作手册。正常情况下，只依靠正文、代码、实验、答案和验收表，就能从零完成阶段2正常掌握要求。

正文提供：

- GEMM从naive、shared tiling、register tiling到Tensor Core的完整推导。
- 不规则shape、低精度、alignment、fallback和多实现调度。
- NSYS、NCU、Roofline、PTX/SASS和编译报告的最低必要读法。
- PyTorch/cuBLAS、手写CUDA和Triton的同协议对照。
- Softmax三遍、片上融合和online状态合并。
- 正确性测试、shape matrix、benchmark、profiling和性能报告模板。
- “优化变慢”实验、课后题、答案和阶段2逐项验收表。
- LayerNorm/RMSNorm、RoPE/SwiGLU、Attention IO、KV Cache、量化、Epilogue和Dispatch扩展教材。
- 最终问题逐题答案和“答案必须来自哪个实验”的对应关系。

只有三类内容需要查询目标环境的官方信息：

1. GPU的精确峰值算力、显存带宽、SM资源和支持的数据类型。
2. 当前CUDA、PyTorch、Triton和Profiler版本对应的安装/指标名称。
3. 不同GPU代际的Tensor Core指令与shape约束。

这些是环境参数，不是缺失课程。本教材会说明要查哪个值，以及这个值怎样进入分析。

### 每章的学习循环

```text
先写性能预测
    ↓
只改变一个主要变量
    ↓
先过正确性，再做benchmark
    ↓
NSYS判断问题在哪一层
    ↓
NCU验证Kernel内部假设
    ↓
写结论、代价和适用边界
```

已掌握的章节可以直接做章末闭卷自查和必做实验；两者都通过就记录证据并跳过。不能用“看过类似代码”代替实验结果。

## 双维学习总入口：概念体系与性能因果关系

阶段 2 不以“记住多少优化技巧”为目标，而以两种能力为目标：

1. **基础概念维度**：知道每个性能概念的定义、计算方式、适用条件和证据来源。
2. **概念关系维度**：能从代码变化推导工作量、访存、资源、调度、指标和 Duration 的变化，并识别其中的取舍与瓶颈迁移。

如果只记得“Shared Tiling 会变快”“向量化能提速”，仍是技巧级理解；能说清在什么 shape、通过什么中间量、用什么指标证明、为什么也可能变慢，才达到阶段 2。

### 维度一：阶段 2 基础概念地图

| 概念组     | 基础概念                                             | 它首先回答的问题                                 | 正文出口         |
| :--------- | :--------------------------------------------------- | :----------------------------------------------- | :--------------- |
| 实验契约   | Shape、dtype、layout、精度、Baseline、统计协议       | 比较的版本是否真的完成同一个任务？               | 第 3、17、18 章  |
| 有效工作量 | Effective FLOPs、最小 bytes、实际指令、实际传输      | “完成任务所需工作”与“硬件实际做的工作”有何区别？ | 第 4、9 章       |
| 结果指标   | Duration、Latency、Throughput、TFLOP/s、有效 GB/s    | 这次优化最终是否让固定任务更快？                 | 第 3、9、18 章   |
| 性能上限   | Arithmetic Intensity、峰值算力、带宽、Roofline       | 在理想条件下更可能受计算还是数据搬运限制？       | 第 4、9 章       |
| 并行映射   | Grid、CTA、Warp、Lane、Wave、Tail、Tile 利用率       | Shape 怎样映射到硬件并行工作和浪费？             | 第 4、5、7 章    |
| 数据复用   | Global、Shared、Register、CTA tile、Thread tile      | 怎样用片上存储减少重复 Global 访问？             | 第 5、6 章       |
| 访存效率   | Coalescing、Alignment、Vectorization、Sector、Bank   | 有效 bytes 怎样变成实际内存请求与传输？          | 第 5、7、9 章    |
| 资源限制   | Registers/thread、Shared/CTA、Threads/CTA、Occupancy | 一个 SM 能同时保留多少工作？                     | 第 5、6、9 章    |
| 调度状态   | Active Warp、Eligible Warp、Issue Slot、Stall        | 驻留的 Warp 是否真的能持续发射指令？             | 第 6、9 章       |
| 重叠与流水 | Load/Compute overlap、Stage、Barrier、依赖链         | 数据等待能否被别的工作或计算隐藏？               | 第 5、6、8 章    |
| 低精度路径 | FP16、BF16、TF32、FP32 accumulate、Tensor Core、WMMA | 精度、布局和 Shape 怎样决定矩阵指令路径？        | 第 8 章          |
| 工具证据链 | NSYS、NCU、Roofline、PTX、SASS、编译资源报告         | 应该在哪一层观察和验证性能假设？                 | 第 9、18 章      |
| 算子迁移   | GEMM、Softmax、Norm、RoPE、Attention、KV Cache、量化 | 同一分析方法怎样迁移到不同计算和访存结构？       | 第 11、26～29 章 |
| 生产工程   | CUDA、Triton、CUTLASS、Dispatch、Autotune、Fallback  | 多个实现怎样在不同输入和硬件上安全选择？         | 第 10、30 章     |
| 系统衔接   | Kernel 时间、同步、Launch、端到端延迟、Amdahl 上限   | 单 Kernel 提速为什么不等于推理服务同比例提速？   | 本节、第 9 章    |

每个概念统一用下面的“性能概念卡”记录：

```text
概念名称：
它回答的性能问题：
定义或计算公式：
成立所需条件：
它的上游变量：
它影响的下游变量：
代码中的控制位置：
NSYS/NCU/PTX/SASS中的证据：
最终怎样回到Duration：
反例、代价或失效Shape：
```

阶段 2 的合格线不是会背定义，而是能从上游变量一路解释到最后三项。

### 维度二：阶段 2 核心性能关系地图

#### 关系链 A：一次可信优化的科学闭环

```text
冻结数学与 API 契约
  → 固定 Shape matrix、dtype、精度和 Baseline
  → 计算 Effective FLOPs 与最小 bytes
  → 提出瓶颈预测和可证伪假设
  → 一次只改变一个主要变量
  → 先验证正确性
  → 再测 Duration 与结果指标
  → NSYS 判断时间落在哪一层
  → NCU 判断 Kernel 内部发生了什么
  → 必要时用 PTX/SASS 证明指令路径
  → 写出结论、代价、适用边界和回归用例
```

这条链是全书主干。任何“优化后快了”的结论，如果跳过工作量、正确性、测量协议或对照实验，都还不能成为简历项目证据。

#### 关系链 B：Tile 大小的完整取舍

```text
CTA/Thread Tile 增大
  ├─ 可能提高 Shared/Register 复用 → Global 访问减少
  ├─ 可能提高每线程工作量 → 指令级并行增加
  ├─ 增加 Registers/thread 或 Shared/CTA → Blocks/SM 可能减少
  ├─ 改变 Grid 数量与 Wave Tail → 小 Shape 并行度可能下降
  └─ 增加边界浪费或 Padding → 不规则 Shape 的无效工作增加
          ↓
Eligible Warp、Stall、实际 bytes、实际指令共同变化
          ↓
固定 Shape 的 Duration 决定最终得失
```

因此 Tile 不是越大越好。选择 Tile 的本质是在“复用、并行度、资源、尾部浪费”之间做 Shape 相关的取舍。

#### 关系链 C：从地址变化到实际内存效率

```text
数据 Layout + Lane 到元素的映射
  → 字节地址的连续性与 Alignment
  → 标量或 Vector Load/Store 指令
  → Request/Sector 数量与 Cache 行为
  → 实际 Global/DRAM bytes 和等待时间
  → Eligible Warp、Memory Stall 与 Duration
```

`float4` 主要改变指令宽度和指令条数，并不自动减少算法最小 bytes；Alignment、尾部和实际生成的指令必须共同验证。Padding 也只有在改变目标 Bank 映射或对齐关系时才可能有效。

#### 关系链 D：从复用到算术强度和 Roofline

```text
算子数学定义
  → Effective FLOPs
实现的数据读写方案
  → 最小 bytes 与估计实际 bytes
FLOPs / bytes
  → Arithmetic Intensity
AI 与峰值算力/带宽的交点
  → Roofline 理论上限
实测 TFLOP/s、GB/s、Duration
  → 判断距离上限还有多远
```

Roofline 是宏观上限模型，不替代 NCU 的调度、指令和资源分析。AI 预测“可能更偏计算或带宽”，不能单独证明当前 Kernel 的实际瓶颈。

#### 关系链 E：从资源到调度，而不是只看 Occupancy

```text
Registers/thread + Shared/CTA + Threads/CTA
  → Blocks/SM 与 Active Warp 上限
  → Occupancy
  → 在指令依赖、Barrier、内存等待的约束下形成 Eligible Warp
  → Scheduler 是否有 Warp 可 Issue
  → Stall 分布与 Issue Slot 利用
  → Duration
```

Active 表示 Warp 已驻留，Eligible 才表示当前周期具备发射条件。优化目标不是让 Occupancy 单项最大，而是让固定工作量更快，同时用 Eligible、Stall 和资源变化解释原因。

#### 关系链 F：从低精度到 Tensor Core 路径

```text
输入 dtype 与允许误差
  → 累加 dtype 和数值契约
  → M/N/K、Layout、Leading Dimension、Alignment 约束
  → WMMA/CUTLASS/Triton 配置与编译选择
  → 是否生成并执行目标 Tensor Core 指令
  → 实际吞吐、转换/搬运开销和数值误差
  → 不满足条件时进入安全 Fallback
```

“输入是 FP16”不等于使用 Tensor Core。必须同时证明接口满足约束、指令路径正确、结果精度合格，并在不规则 Shape 上保留安全路径。

#### 关系链 G：Profiler 的分层证据

```text
端到端或 Benchmark 变慢
  → NSYS：CPU、Launch、Memcpy、同步还是 Kernel 时间？
  → 若是 Kernel：NCU 看 Duration、Launch、SM/Memory、DRAM、Occupancy、Scheduler/Stall
  → 若指令路径有疑问：PTX/SASS 或编译报告确认
  → 回到源代码中能控制的 Tile、地址、资源、同步或 Dispatch
  → 重跑同协议 Duration
```

工具的关系是“先分层、再下钻、最后回到结果”，不是把所有指标同时抄进报告。每个指标都必须对应一个待验证假设。

#### 关系链 H：优化后的瓶颈迁移

```text
原瓶颈：重复 Global 读取
  → Shared Tiling 减少读取
  → 内存等待下降
  → 新瓶颈可能变成同步、Shared 带宽、指令依赖或资源限制
  → 再次测量并建立下一轮假设
```

这是原文需要集中补强的关系：优化不是消灭所有瓶颈，而是把限制推到下一层。每完成一个版本，都要问“旧瓶颈是否真的下降，新瓶颈迁移到了哪里”，不能永久沿用第一轮结论。

#### 关系链 I：同一方法怎样迁移到不同算子

```text
统一入口：数学契约 → Shape → FLOPs/bytes → 并行映射 → 片上状态 → 同步 → 数值 → 证据
  ├─ GEMM：重点是二维/三维 Tile、复用和矩阵指令
  ├─ Softmax：重点是行归约、数值稳定和多遍读写
  ├─ Norm：重点是统计量、稳定算法与融合
  ├─ RoPE/SwiGLU：重点是逐元素访问和 Epilogue 融合
  ├─ Attention：重点是中间矩阵 IO、Online 状态与 KV 访问
  └─ 量化：重点是 Scale/Zero-point、反量化开销和低精度指令
```

迁移不是照搬 GEMM Tile，而是保留分析问题的顺序，再根据算子的数据依赖重新选择映射和片上状态。

#### 关系链 J：从单 Kernel 到推理系统收益

```text
目标 Kernel 的原耗时占比 p
  + Kernel 自身加速比 s
  → Amdahl 上限：整体加速比 ≤ 1 / ((1-p) + p/s)
  → 再叠加 Launch、同步、排队、Memcpy、其他算子和服务调度
  → 最终 TTFT、TPOT、吞吐或端到端延迟变化
```

例如某 Kernel 占端到端时间 20%，即使自身无限快，整体理论加速也不超过 `1 / 0.8 = 1.25×`。这条关系用于衔接你后续的推理系统方向：Kernel Duration 是必要证据，但系统价值还取决于调用频率、时间占比和是否引入额外同步或数据搬运。

#### 关系链 K：从多个实现到生产 Dispatch

```text
输入 Shape/dtype/layout/alignment + GPU 架构
  → 判断实现约束是否满足
  → Dispatch 到专用快路径或通用 Fallback
  → Autotune 在代表性 Shape 上选择配置
  → 正确性回归 + 性能回归
  → 新 Shape 或新 GPU 触发重新评估
```

单一 Shape 上最快的 Kernel 不等于生产可用。阶段 2 最终要能解释“谁选择实现、选择依据是什么、约束不满足怎么办、性能退化怎样被发现”。

### 五种关系必须同时掌握

| 关系类型 | 要回答的问题                           | 合格示例                                                         |
| :------- | :------------------------------------- | :--------------------------------------------------------------- |
| 计算关系 | 指标怎样由 Shape、FLOPs、bytes 得到？  | GEMM Effective FLOPs 为 `2MNK`，有效 TFLOP/s 由它除以 Duration   |
| 映射关系 | 数学维度怎样映射到 Tile、Warp 和地址？ | M/N 映射输出 Tile，K 映射分段累加与数据复用                      |
| 因果关系 | 代码变化通过哪些中间量影响 Duration？  | Thread Tile 增大 → Shared 读取减少、寄存器增加 → 调度变化 → 时间 |
| 取舍关系 | 收益同时带来了什么成本？               | 多级流水可能隐藏访存，也增加 Shared/寄存器、同步复杂度和尾部成本 |
| 边界关系 | 结论在哪些 Shape、dtype 或硬件上失效？ | Tensor Core 快路径受 Layout、Alignment、K 维和架构支持约束       |

### 双维闭卷自查：不要只回答“是什么”

学习任一优化点后，关闭教材回答下面七项：

1. 这个概念或技术解决哪个明确的性能问题？
2. 它的定义、公式或硬件作用对象是什么？
3. 它的上游控制变量位于 Shape、Layout 还是代码配置的哪里？
4. 它通过哪些中间量影响实际执行和 Duration？
5. 至少两个竞争假设是什么，怎样分别证伪？
6. 用哪一层工具、哪个指标或哪段指令证明？
7. 它的代价、失效 Shape、Fallback 和回归用例是什么？

例如，不能只回答“向量化能减少指令”，而要形成完整回答：

```text
地址连续且满足Alignment
  → 编译器生成宽Load/Store
  → 每个线程的内存指令数可能减少
  → 但算法最小bytes不变，尾部还需标量Fallback
  → 检查SASS指令、request/sector、Duration与不规则Shape正确性
```

七项中任何一项无法回答，就按关系链回到对应前置章节。后文 25 道正式口试题仍是阶段出口；本节负责组织知识和暴露断点，不额外增加无限学习清单。

---

## 0. 阶段2完成后，你能承担什么工作

你将能够接手一个真实算子的性能任务：

```text
收到shape、dtype和性能目标
  ├─ 定义输入、精度和错误契约
  ├─ 建立PyTorch/供应商库与朴素基线
  ├─ 估算FLOPs、bytes和性能上限
  ├─ 设计线程、warp、CTA和tile映射
  ├─ 实现多个版本并处理不规则shape
  ├─ 用NSYS/NCU判断瓶颈和退化原因
  ├─ 与Triton或CUTLASS做同协议对照
  └─ 输出可复现的性能报告与回归测试
```

### 主交付 A：GEMM旗舰项目

实现 `C[M,N] = A[M,K] × B[K,N]`，至少包含：

1. FP32 naive。
2. FP32 shared-memory tiled。
3. FP32 register tile 2×2。
4. register tile 4×4退化或改善对照。
5. padding和vectorized access受控实验。
6. FP16 Tensor Core/WMMA，FP32累加。
7. FP16/BF16 Triton实现，FP32累加。
8. 非tile整数倍shape的安全路径。
9. PyTorch/cuBLAS基线、完整shape matrix和性能报告。

### 迁移交付 B：Softmax案例

实现二维Tensor最后一维Softmax，至少包含：

1. PyTorch组合式三遍基线。
2. 一个block处理一行的片上融合CUDA版本。
3. online `(m, l)` 状态合并版本或完整推导和验证。
4. FP32、FP16/BF16输入，FP32统计量。
5. 短行、长行、非2次幂、mask/scale和极值测试。
6. 有效GB/s、读写次数和一次NCU分析。

### 完成标准

- **正确**：正常、边界、真实、压力shape和错误输入全部有证据。
- **可解释**：能从地址和资源推导指标方向，不靠背结论。
- **可复现**：固定环境、输入、warmup、统计和基线。
- **能证伪**：优化变慢时至少提出两个竞争假设并逐个排除。
- **可迁移**：GEMM方法能迁移到Softmax，不是只复述一个教程。

---

## 1. 阶段2总问题树

```text
一个Kernel为什么快或慢？
  ├─ 做的工作是否相同？
  │   ├─ FLOPs是否改变？
  │   ├─ bytes是否改变？
  │   └─ 精度、输出和边界是否相同？
  ├─ GPU是否得到足够工作？
  │   ├─ grid是否太小？
  │   ├─ wave tail是否严重？
  │   └─ shape是否让tile大量浪费？
  ├─ 数据是否高效到达计算单元？
  │   ├─ global访问是否合并？
  │   ├─ 数据是否被shared/register复用？
  │   ├─ 是否有多余request/sector？
  │   └─ 是否发生bank conflict或spill？
  ├─ 计算单元是否高效执行？
  │   ├─ SIMT还是Tensor Core路径？
  │   ├─ 指令依赖链是否过长？
  │   └─ eligible warp是否足够？
  ├─ 资源是否限制并行度？
  │   ├─ registers/thread
  │   ├─ shared memory/block
  │   └─ threads/block与blocks/SM
  └─ 测量是否可信？
      ├─ CPU/launch/sync问题先由NSYS判断
      ├─ Kernel内部问题由NCU判断
      └─ 所有结论最终回到Duration
```

**一句话记忆：先确认工作量相同，再解释硬件如何完成这些工作。**

---

## 2. 最短学习顺序

| 顺序  | 教材模块                      | 立即产出                       | 通过证据                        |
| :---: | :---------------------------- | :----------------------------- | :------------------------------ |
| 1     | 性能实验协议与可信基线        | 环境快照、shape matrix、基线表 | 同一输入重复结果稳定            |
| 2     | GEMM数学、naive映射和性能模型 | naive kernel与手写FLOPs/bytes  | 推导与实测方向一致              |
| 3     | Shared tiling                 | tiled kernel                   | global读取复用和同步点可解释    |
| 4     | Register tiling与资源取舍     | 2×2、4×4版本                   | 能解释变快或变慢                |
| 5     | Padding、向量化与不规则shape  | 三个受控实验和安全fallback     | 快路径和fallback均有测试        |
| 6     | 低精度与Tensor Core           | FP16 WMMA、BF16/FP16 Triton    | 指令路径、精度和shape约束有证据 |
| 7     | NSYS、NCU、Roofline和指令验证 | 一份完整Profiler因果报告       | 八步树能落到代码和Duration      |
| 8     | Triton同协议对照              | 可修改tile/config的实现        | 与CUDA/cuBLAS公平比较           |
| 9     | Softmax方法迁移               | 第二算子案例                   | 读写、归约、精度和瓶颈闭环      |
| 10    | 报告、回归与闭卷复测          | 主报告、失败案例、性能回归     | 一周后不看教程仍能重写核心版本  |

建议投入60～90小时。每周8～10小时通常需要8～10周；已经完成过相关实验时，以验收证据为准，不机械服从周数。

---

## 3. 模块1：先建立不会自欺的性能实验

### 3.1 这个模块解决什么问题

你必须先保证不同版本执行的是同一个任务。否则“更快”可能只是少算、低精度、漏边界、隐式转换或错误计时。

### 3.2 GEMM统一契约

阶段2的统一接口：

```text
输入A：[M,K]，row-major，CUDA contiguous
输入B：[K,N]，row-major，CUDA contiguous
输出C：[M,N]
数学定义：C[m,n] = Σ A[m,k] × B[k,n]
```

必须明确：

- FP32路径：FP32输入、FP32累加、FP32输出。
- FP16/BF16路径：低精度输入、FP32累加；输出dtype由接口显式规定。
- 不支持的stride明确拒绝，不偷偷复制后把复制时间排除。
- M/N/K为0时定义输出shape和是否launch。
- 不规则M/N/K必须正确，Tensor Core精确路径不满足条件时安全fallback。
- 所有版本使用PyTorch当前device和当前stream。

### 3.3 Shape matrix

| 类别       | M    | N     | K     | 为什么必须测             |
| :--------- | ---: | ----: | ----: | :----------------------- |
| 小方阵     | 128  | 128   | 128   | launch、并行度和固定开销 |
| 中方阵     | 1024 | 1024  | 1024  | 基本吞吐比较             |
| 大方阵     | 4096 | 4096  | 4096  | 计算路径和稳定吞吐       |
| 小M投影    | 1    | 4096  | 4096  | decode式低并行度         |
| 长条M      | 128  | 4096  | 4096  | 小batch/prefill边界      |
| 长条N      | 4096 | 11008 | 4096  | FFN上投影形态            |
| 长条K      | 4096 | 4096  | 11008 | FFN下投影形态            |
| 不规则     | 123  | 145   | 67    | M/N/K尾部同时存在        |
| tile边界前 | 255  | 257   | 511   | 非整除与wave/tile浪费    |
| 空维度     | 0/1  | 0/1   | 0/1   | API语义和零launch        |

真实GPU显存不足时按比例缩小大shape，但必须保留方阵、小M、长条和不规则四类结构。

### 3.4 基线分层

每个版本至少与三种基线比较：

1. **数学reference**：CPU或高精度小shape，验证定义。
2. **PyTorch `torch.matmul`**：真实框架基线，通常落到供应商库。
3. **自己的naive**：解释每一步优化相对什么发生变化。

不能把自己的教学kernel比naive快很多，就宣传为“高性能GEMM”；与cuBLAS差距必须诚实保留。

### 3.5 最小测量协议

固定：GPU、驱动、CUDA、PyTorch、编译参数、shape、dtype、输入seed、时钟/功耗状态、warmup、repeats和同步位置。

报告：median、P90、P95、min/max、有效TFLOP/s；访存算子再报告有效GB/s。

```text
GEMM FLOPs = 2 × M × N × K
TFLOP/s = FLOPs / time_seconds / 10^12
```

这里的TFLOP/s是“按数学有效工作量计算的吞吐”，不是GPU执行的所有指令数。

### 3.6 闭卷自查

- [ ] 所有版本的数学定义、dtype、累加和输出协议相同。
- [ ] shape matrix同时覆盖真实、边界、不规则和压力输入。
- [ ] 每次实验只改变一个主要变量，运行前写预测。
- [ ] 同时保留naive、PyTorch/cuBLAS和当前最佳版本。
- [ ] 性能报告有统计分布，不用单次最好成绩。

### 3.7 停止线

本阶段不建设跨机器性能数据库，不锁死某个GPU频率作为唯一真相。能保证同一环境中的受控对照可信即可。

---

## 4. 模块2：GEMM数学、Naive映射和性能模型

### 4.1 这个模块解决什么问题

写代码前先回答：每个输出要做多少计算、读多少数据、线程负责什么、理论瓶颈方向是什么。

### 4.2 从数学到地址

```text
C[M,N] = A[M,K] × B[K,N]
C[m,n] = Σ(k=0..K-1) A[m,k] × B[k,n]
```

row-major地址：

```text
A[m,k]地址偏移 = m × K + k
B[k,n]地址偏移 = k × N + n
C[m,n]地址偏移 = m × N + n
```

Naive映射让一个thread负责一个`C[m,n]`：

```text
row = blockIdx.y × blockDim.y + threadIdx.y
col = blockIdx.x × blockDim.x + threadIdx.x
```

每个有效thread执行：

- K次读取A。
- K次读取B。
- K次FMA，按2 FLOPs计为2K FLOPs。
- 1次写C。

### 4.3 Naive的理论流量

从单个输出thread看，FP32最低代码级流量近似：

```text
读A：4K bytes
读B：4K bytes
写C：4 bytes
总计：8K + 4 bytes
算术强度：2K / (8K + 4) ≈ 0.25 FLOPs/byte
```

这是“没有跨thread cache复用”的朴素上界模型。真实L1/L2会复用部分数据，所以Profiler实测DRAM bytes可能更少；模型的价值是给出假设，不是替代测量。

整个数学问题理想最低DRAM流量是每个A、B只读一次、C只写一次：

```text
4 × (M×K + K×N + M×N) bytes
```

naive代码级重复读取与理想最低流量之间的巨大差距，就是tiling要解决的问题。

### 4.4 线程访问模式

同一warp若thread按相邻`col`排列：

- 对固定k，B[k,col]通常是相邻地址，利于合并。
- 对固定row，不同thread读取相同A[row,k]，可能被cache/broadcast复用。
- 一个thread内部的K次FMA形成accumulator依赖链。

因此“naive一定完全不合并”是错误说法。它的问题是数据复用依赖cache、每个输出重复读取多、计算与数据搬运比例低。

### 4.5 必做手算

给定`M=N=K=1024`、block为`16×16`：

1. grid维度。
2. block和thread总数。
3. 每thread的FMA和FLOPs。
4. 全部有效数学FLOPs。
5. naive代码级读取量与理想最低bytes。
6. 相邻thread读取A/B的地址关系。

答案在第19章。

### 4.6 闭卷自查

- [ ] 能从M/N/K推导索引、FLOPs和最低bytes。
- [ ] 能区分代码级load、cache请求和DRAM实际bytes。
- [ ] 能解释naive中A与B各自的warp访问模式。
- [ ] 能说明为什么先写模型、再用Profiler修正。

### 4.7 停止线

暂不追求精确模拟每级cache命中。阶段2需要的是可验证的一阶模型，不是GPU周期模拟器。

---

## 5. 模块3：Shared-memory Tiling

### 5.1 这个模块解决什么问题

让一个block协作加载A/B tile，复用片上数据，减少多个输出thread对global memory的重复读取。

### 5.2 CTA tile映射

以`TILE=16`为例：

```text
一个block：16×16 = 256 threads
一个block输出：C的16×16 tile
每个thread输出：C tile中的1个元素
K维：每次处理16，循环ceil(K/16)次
```

每个K tile：

1. 256个thread协作加载A的16×16 tile。
2. 协作加载B的16×16 tile。
3. block barrier，保证数据到shared。
4. 每thread使用shared中的一行A和一列B做16次FMA。
5. block barrier，保证所有thread用完旧tile再覆盖shared。

### 5.3 全局读取减少倍数

在一个16×16输出tile、一个K分块中：

```text
naive：256个输出 × (16个A + 16个B) = 8192个元素读取
tiled：A tile 256 + B tile 256 = 512个元素读取
理论代码级读取减少：8192 / 512 = 16倍
```

减少的是重复global load指令/请求机会；实际DRAM减少倍数受cache影响，不能直接声称也是16倍。

### 5.4 为什么有两个barrier

- 第一个barrier保护“shared写完后再读”。
- 第二个barrier保护“所有thread读完后再被下一轮覆盖”。

边界thread不能在barrier前return。越界元素应向shared写0作为乘加identity，然后所有thread共同到达barrier。

### 5.5 Shared占用和bank

两个`16×16 FP32` tile：

```text
shared/block = 2 × 16 × 16 × 4 = 2048 bytes
```

bank conflict必须根据真实访问方向判断。`As[ty][k]`与`Bs[k][tx]`可能包含广播或连续访问；盲目给所有二维数组加`+1` padding，可能只增加shared占用而没有收益。

### 5.6 Tile选择

tile增大可能：

- 提高复用、减少K循环和barrier次数。
- 增加threads/block或每thread工作。
- 增加shared和register。
- 增加M/N尾部浪费。
- 降低blocks/SM或让小shape并行度不足。

因此tile不是越大越好。阶段2至少比较8、16、32中的两个合法设计，但每个版本必须保持其他主要变量尽量一致。

### 5.7 必做实验

比较naive与TILE16：

1. 运行前计算理论global读取变化、shared用量和barrier次数。
2. 正确性覆盖K=15/16/17与M/N尾部。
3. Benchmark方阵、小M和不规则shape。
4. NCU比较Duration、global requests/sectors、DRAM bytes、shared、occupancy和stall。
5. 写出“模型正确的部分”和“被cache/硬件修正的部分”。

### 5.8 闭卷自查

- [ ] 能画出CTA tile、thread输出和K tile循环。
- [ ] 能推导global读取减少倍数，而不是背“shared更快”。
- [ ] 能解释两个barrier分别保护什么。
- [ ] 能处理M/N/K非tile整数倍。
- [ ] 能根据地址判断padding是否可能有效。

### 5.9 停止线

暂不学习异步拷贝、TMA和复杂swizzle。先把同步版tiled GEMM的复用模型与实测因果做扎实。

---

## 6. 模块4：Register Tiling与资源取舍

### 6.1 这个模块解决什么问题

Shared tiling后，A/B已在片上，但每个thread只计算一个输出，仍会重复从shared读取。Register tiling让每个thread计算多个输出，复用刚读入的A/B值。

### 6.2 2×2 thread tile

```text
一个thread输出2×2 = 4个C元素
需要4个FP32 accumulator
每个K步读取2个A值和2个B值
完成4次FMA
```

如果4个输出分别由4个thread计算，每个K步合计需要8个shared值；合并到一个thread后只需4个，shared读取复用提高。

代价：

- accumulator从1个增加到4个。
- 临时A/B值增加。
- 地址和边界逻辑增加。
- registers/thread上升，可能减少blocks/SM。
- 一个thread tile覆盖更大输出，grid block数减少，小shape可能不够并行。

### 6.3 4×4为什么可能更慢

4×4每thread有16个accumulator，还需要A/B临时数组。它可能提高shared复用，也可能：

1. registers/thread显著增加。
2. blocks/SM下降。
3. eligible warps减少，延迟隐藏变差。
4. 出现spill/local memory。
5. grid过小或尾部浪费增加。
6. 指令依赖和地址计算增加。

不能看到register增加就直接判定根因。必须同时检查register、spill、occupancy、active/eligible warps、stall、grid规模和Duration。

### 6.4 竞争假设实验

若4×4慢于2×2，至少提出：

- H1：寄存器压力降低了blocks/SM和eligible warps。
- H2：没有spill，真正原因是grid block数减少导致wave利用差。
- H3：不规则shape让4×4 tile浪费更多计算和分支。
- H4：shared访问或指令依赖变差。

逐次只改变shape、thread tile或编译寄存器限制中的一个变量。最后必须能排除至少一个看似合理但错误的假设。

### 6.5 必做实验

1. 实现2×2和4×4，保持K tile相同。
2. 编译时加入`--ptxas-options=-v`记录register和spill。
3. 对方阵、小M、不规则shape比较grid、waves和Duration。
4. NCU观察occupancy、eligible warps、主要stall和local memory。
5. 不论4×4快慢，都写出适用shape和代价。

### 6.6 闭卷自查

- [ ] 能推导thread tile怎样减少shared读取。
- [ ] 能画出CTA tile、thread tile和输出地址。
- [ ] 能从register变化推到blocks/SM，但不把occupancy当结论。
- [ ] 优化变慢时能提出竞争假设和最小实验。

### 6.7 停止线

阶段2不要求手写达到cuBLAS级warp-specialized pipeline。2×2/4×4足以训练资源权衡和反直觉分析。

---

## 7. 模块5：Padding、Vectorization与不规则Shape

### 7.1 Padding

Padding只在真实访问发生bank conflict时有意义。常见32 banks、4-byte模型：

```text
bank = (byte_address / 4) % 32
```

若warp沿二维数组的列访问，leading dimension为32时可能全部落到同一bank；改成33可让bank轮转。若访问本来是连续行或同地址广播，padding可能没有收益。

必做：对tiled与tiled+padding使用完全相同的映射，先手算bank，再比较shared conflict绝对量与Duration。允许结论是“padding无效”，但必须说明为什么。

### 7.2 Vectorized access

`float4`/`half2`可能减少load/store指令并扩大单指令搬运宽度，但不会自动减少数学bytes。

快路径条件：

- 基地址满足类型alignment。
- 每行leading dimension保证后续行仍对齐。
- 向量覆盖范围不越界。
- 尾部有标量或masked fallback。
- 输出地址也满足store alignment。

必须比较：指令数、requests、sectors、实际bytes和Duration。若只把4个scalar load换成一个vector load，但瓶颈在别处，时间可以不变。

### 7.3 不规则Shape策略

三种常见方案：

1. kernel内逐元素mask：通用，分支和无效工作较多。
2. 主体tile + tail kernel：快路径干净，代码和launch增加。
3. 多实现调度：整齐shape用专用快路径，其余走通用fallback。

阶段2至少实现第3种思想：WMMA只处理满足约束的shape；其余明确进入通用路径。不能越界，也不能静默改变dtype。

### 7.4 Shape浪费率

对输出tile `TM×TN`：

```text
launched_output_slots
= ceil(M/TM) × TM × ceil(N/TN) × TN

有效利用率
= M × N / launched_output_slots
```

K维还要乘上`K / (ceil(K/TK)×TK)`的有效比例。小M、奇数维度和tile边界附近必须计算浪费率。

### 7.5 闭卷自查

- [ ] 能从地址公式判断padding是否必要。
- [ ] 能解释vectorization改变的是指令、请求还是bytes。
- [ ] 能设计alignment检查和tail fallback。
- [ ] 能计算不规则shape的tile有效利用率。
- [ ] 多实现调度不会静默改变输入、输出和精度契约。

---

## 8. 模块6：低精度、Tensor Core和WMMA

### 8.1 这个模块解决什么问题

理解低精度输入怎样进入Tensor Core、为什么累加通常使用FP32、shape/layout为何受限，以及怎样证明实际生成了MMA指令。

### 8.2 三层tile

```text
CTA tile：一个block负责的C区域
  └─ warp tile：一个warp负责的子区域
      └─ MMA tile：一条或一组Tensor Core指令的矩阵形状
```

WMMA常用教学形态是`m16n16k16`：一个warp协作完成16×16输出tile的一次K=16矩阵乘加。fragment是warp协作持有的逻辑矩阵片段，元素具体分散在哪个lane由实现规定，不能把它当普通每thread数组解释。

### 8.3 数据类型

- FP16输入 + FP32 accumulator：范围和累加误差通常优于FP16累加。
- BF16输入 + FP32 accumulator：动态范围接近FP32，精度低于FP16。
- TF32：通常用于FP32输入的Tensor Core内部计算路径，不是普通Tensor存储dtype。
- 输出FP16/BF16会再发生一次舍入；输出FP32占用更多带宽。

阶段2的WMMA参考实现使用FP16输入、FP32输出。BF16通过Triton `tl.dot`路径完成，并用Profiler确认Tensor Core/MMA指令；低层BF16 WMMA接口随架构和工具链差异较大，不作为第一份手写实现的必选项。

### 8.4 Layout和leading dimension

WMMA fragment声明A/B是row-major或col-major；`load_matrix_sync`的leading dimension必须与实际内存布局一致。layout声明错时，结果可能像“随机数”，但根因不是精度。

所有参与warp的lane必须以一致控制流执行WMMA操作。M/N/K不满足tile约束时，不能让warp中部分lane跳过MMA；应由launcher路由到fallback。

### 8.5 怎样证明用了Tensor Core

证据强度从弱到强：

1. 代码调用WMMA或`tl.dot`：只是意图。
2. 编译报告和NCU Tensor Core相关吞吐：支持证据。
3. SASS/PTX中出现对应MMA/HMMA指令：直接路径证据。
4. 与SIMT FP32版本的受控性能和精度差异：结果证据。

不能只因FP16版本更快就断言使用了Tensor Core；更少bytes本身也可能带来收益。

### 8.6 安全fallback

```text
if dtype、layout、alignment、M/N/K约束全部满足：
    走WMMA/Tensor Core快路径
else：
    走通用CUDA或供应商库fallback
```

报告必须把fallback转换、额外分配和launch算入端到端时间；如果只比较核心kernel，要单独标注不含哪些成本。

### 8.7 闭卷自查

- [ ] 能画出CTA/warp/MMA tile层级。
- [ ] 能解释fragment、layout、leading dimension和warp一致执行。
- [ ] 能说明FP16/BF16/TF32的存储、累加和误差差异。
- [ ] 能用指令或Profiler证据确认Tensor Core路径。
- [ ] 非整齐shape有安全fallback并计入成本。

---

## 9. 模块7：NSYS、NCU、Roofline和指令验证

### 9.1 工具分工

| 工具          | 回答的问题                                             | 不应直接回答的问题               |
| :------------ | :----------------------------------------------------- | :------------------------------- |
| CPU wall time | Python到GPU完成的端到端延迟                            | 单个Kernel内部瓶颈               |
| CUDA Event    | 同stream上GPU elapsed time                             | CPU gap和跨进程服务延迟          |
| NSYS          | 时间线、CPU gap、launch、同步、拷贝、Kernel排列        | 单个Kernel具体stall根因          |
| NCU           | Kernel资源、访存、scheduler、stall、Roofline和源码热点 | 完整请求链路和多Kernel空隙       |
| 编译报告      | registers、shared、stack和spill                        | 这些资源是否真的导致Duration变化 |
| PTX/SASS      | vector、MMA、load/store等指令路径                      | 指令为何一定是最终瓶颈           |

### 9.2 先用NSYS分层

先判断时间花在：

- Python/C++准备。
- dtype转换或临时分配。
- Kernel launch gap。
- H2D/D2H拷贝。
- 强制同步。
- 目标Kernel本身。

如果小shape主要受launch影响，直接优化Kernel内部可能对端到端几乎没有收益。

### 9.3 NCU八步问题树

固定顺序：

```text
1. Duration
2. Launch Stats
3. SM vs Memory方向
4. DRAM/L2/L1/shared绝对bytes、requests、sectors
5. Occupancy和资源限制
6. Scheduler与eligible warps
7. Stall与源码/指令位置
8. Roofline作为宏观交叉检查，再回到Duration
```

注意：Roofline不是自动给根因的最终裁判；先确认FLOPs、bytes、dtype和内存层级选对。

### 9.4 怎样读stall

先问scheduler是否经常没有eligible warp。只有issue slot明显空闲时，stall分类才值得成为主线。

- long scoreboard常与较长延迟数据依赖有关，但要回到对应load和cache层级。
- short scoreboard可能与较短延迟依赖/shared操作有关。
- barrier提示warp在等同步，但可能是负载不均而非barrier指令本身“慢”。
- math pipe throttle提示某计算管线饱和，但要确认执行的是有效工作还是尾部浪费。
- not selected不自动代表问题，可能只是有其他eligible warp被选中。

### 9.5 Throughput与总工作量

百分比上升不代表有效工作增加。例如DRAM throughput提高，可能因为：

- 时间缩短、bytes相同，这是好事。
- 实际bytes增加、时间相同，这可能是退化。
- cache命中变化导致流量层级转移。

每次同时看百分比、绝对bytes、requests/sectors和Duration。

### 9.6 Roofline

```text
Arithmetic Intensity = 有效FLOPs / 选定内存层级bytes
Ridge Point = 峰值FLOP/s / 峰值bandwidth
```

点位于ridge左侧通常偏带宽限制，右侧通常偏计算限制。但实际实现还可能低于roofline很多，原因包括并行度、指令依赖、资源、launch和非理想访问。

### 9.7 因果报告句式

```text
变化：2×2改成4×4，每thread输出从4增到16
预测：shared读取复用提高，但register和tile浪费增加
证据：register X→Y，blocks/SM A→B，eligible warp下降；无spill
排除：DRAM bytes基本不变，不支持“global流量增加”假设
结果：方阵快P%，小M慢Q%
结论：4×4收益依赖足够grid与规则shape，退化主因是并行度/资源而非spill
代价：更复杂边界和更窄适用范围
```

### 9.8 闭卷自查

- [ ] 先用NSYS判断层级，再用NCU分析Kernel。
- [ ] NCU按八步顺序，不从某个百分比直接跳结论。
- [ ] stall与eligible warp、源码和资源联动。
- [ ] Roofline中的FLOPs、bytes、dtype和层级有明确定义。
- [ ] 所有结论最终回到Duration和正确性。

---

## 10. 模块8：Triton/CUTLASS分层与同协议对照

### 10.1 为什么阶段2要学一个主流栈

手写CUDA训练硬件映射；Triton/CUTLASS展示工业开发中怎样用更高抽象复用调度、tile和Tensor Core能力。目标不是多学一个语法，而是比较：

- 开发效率。
- 控制粒度。
- 可移植性。
- 自动调优能力。
- 对特殊融合和不规则shape的适应性。

### 10.2 分层

```text
PyTorch / torch.compile
    ↓ 图与算子层
Triton
    ↓ program、block pointer、tl.dot，编译器生成GPU代码
CUTLASS / CuTe
    ↓ C++模板、layout algebra、copy/MMA atom、GEMM hierarchy
CUDA C++ / WMMA
    ↓ thread/warp/shared/register与较直接硬件控制
PTX / SASS
    ↓ 指令层
```

阶段2选择Triton完成一条非教程式实现：必须修改tile/config或调度，并使用自己的shape matrix、容差和统计协议。CUTLASS只要求理解定位，不同时展开两套大型API。

### 10.3 公平对照

- 相同A/B、shape、dtype和输出dtype。
- 相同累加精度和容差。
- 相同warmup/repeats与计时器。
- 说明是否包含编译、autotune、dtype转换和临时分配。
- 首次JIT时间与稳态执行时间分开。
- cuBLAS、Triton和手写CUDA都保留，不挑选有利shape隐藏失败。

### 10.4 闭卷自查

- [ ] 能解释Triton、CUTLASS/CuTe、WMMA和CUDA C++的抽象层次。
- [ ] Triton实现不是原样照抄，能修改BLOCK_M/N/K、num_warps或grouping。
- [ ] JIT/autotune成本与稳态时间分开。
- [ ] 所有实现使用同一协议并诚实解释差距。

---

## 11. 模块9：把方法迁移到Softmax

### 11.1 为什么选择Softmax

Softmax包含数值稳定、两次reduction、指数、读写融合和长短行映射，能验证你是否真正掌握访存、同步、精度和性能模型。

### 11.2 数学定义

```text
m = max(x)
l = Σ exp(x_i - m)
y_i = exp(x_i - m) / l
```

减去max避免指数溢出，不改变数学结果。

### 11.3 三种实现

#### PyTorch组合式三遍

`max → subtract/exp → sum → divide`由多个算子组成，中间Tensor写回global memory并产生多个launch。它是数学基线，不是理想性能实现。

#### 一个block一行的片上融合

把一行加载到shared/register，block内完成max和sum，再写输出。理论上可接近“读输入一次、写输出一次”，但一行必须适合片上资源；长行可能受shared/register和并行度限制。

#### Online状态合并

对一段数据维护：

```text
m = 当前最大值
l = Σ exp(x_i - m)
```

合并两段`(m1,l1)`和`(m2,l2)`：

```text
m = max(m1, m2)
l = l1 × exp(m1-m) + l2 × exp(m2-m)
```

它允许不同thread/warp独立处理分块后再归约状态。最终输出仍需知道全局m/l，因此普通Softmax至少还要再读一次x；它把统计量计算合并为一遍，并为FlashAttention式分块状态更新打基础。

### 11.4 读写模型

对`rows×cols`：

- 组合式PyTorch会多次读写中间Tensor，实际由框架融合情况决定。
- 三次global循环的单kernel：读x三次、写y一次。
- online统计+输出：读x两次、写y一次。
- 整行片上缓存：理想读x一次、写y一次，但受片上容量限制。

有效GB/s常按最低必要流量计算：

```text
effective_GB/s
= (input_bytes + output_bytes) / time_seconds / 1e9
```

必须标注这是“有效流量”，不等于Profiler观测到的实际DRAM bytes。

### 11.5 映射选择

- 很短行：一个warp一行，减少barrier和闲置。
- 中等行：一个block一行，多个warp合作归约。
- 很长行：thread循环、多个block/分阶段或更复杂持久化设计。
- rows很少：grid不足，单行再快也可能无法占满GPU。

边界不是固定数字，由dtype、cols、寄存器、shared、GPU和实现共同决定；必须通过shape sweep确定。

### 11.6 融合

Attention中常见：

```text
softmax(scale × score + mask)
```

若scale和mask在Softmax kernel中直接应用，可减少中间Tensor读写和launch。收益来自消除数据搬运/launch，不是减少Softmax数学定义。Causal或无效token的mask值应等价于负无穷，并正确处理整行全mask语义。

### 11.7 必做实验

1. FP32/FP16/BF16，统计用FP32。
2. cols=1/31/32/33/127/128/129/1024/4097。
3. rows=1/32/1024/4096，观察并行度。
4. 极大正负数、全相等、NaN/Inf和全mask。
5. 比较PyTorch、CUDA融合和online版本的有效GB/s。
6. NCU判断是DRAM、片上资源、并行度、指数吞吐还是launch限制。

### 11.8 闭卷自查

- [ ] 能推导max trick和online `(m,l)` 合并公式。
- [ ] 能估算三种实现的读写次数。
- [ ] 能根据rows/cols选择warp或block映射并验证边界。
- [ ] 能说明mask/scale融合减少什么，没有减少什么。
- [ ] 能区分有效GB/s与实际DRAM bytes。

---

## 12. 阶段2配套工程结构

阶段1工程继续保留，新建独立目录：

```text
stage2_kernel_lab/
  ├─ setup.py
  ├─ src/
  │   ├─ bindings.cpp
  │   ├─ gemm.cu
  │   └─ softmax.cu
  ├─ python/
  │   └─ triton_gemm.py
  ├─ tests/
  │   ├─ test_gemm.py
  │   └─ test_softmax.py
  ├─ benchmark/
  │   ├─ bench_gemm.py
  │   ├─ bench_softmax.py
  │   └─ profile_entry.py
  ├─ reports/
  │   ├─ environment.md
  │   ├─ version_results.csv
  │   ├─ ncu/
  │   └─ nsys/
  └─ README.md
```

> 验证边界：当前文档编写环境没有安装PyTorch/CUDA，后续参考代码只完成静态结构检查，尚未在本机实际编译。第一次在目标GPU服务器运行时，必须记录编译器、GPU、软件版本、测试结果和必要修正。

后续章节给出参考实现。第一次可以跟写；最终验收时必须关闭参考代码，从空文件独立写出naive、tiled和regtile2x2核心版本。

---

## 13. 构建与绑定

### 13.1 `setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="stage2_kernel_lab",
    ext_modules=[
        CUDAExtension(
            name="stage2_kernel_lab",
            sources=[
                "src/bindings.cpp",
                "src/gemm.cu",
                "src/softmax.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--ptxas-options=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

这里不启用`--use_fast_math`。先建立默认数学路径；以后单独增加fast-math版本，才能把指数近似、FMA、精度和性能作为受控变量比较。

### 13.2 `src/bindings.cpp`

```cpp
#include <torch/extension.h>


torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_tiled(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_tiled_padding(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_regtile2x2(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_regtile4x4(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_float4(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_wmma_fp16(torch::Tensor a, torch::Tensor b);

torch::Tensor softmax_fused(torch::Tensor x);
torch::Tensor softmax_online(torch::Tensor x);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive);
    m.def("gemm_tiled", &gemm_tiled);
    m.def("gemm_tiled_padding", &gemm_tiled_padding);
    m.def("gemm_regtile2x2", &gemm_regtile2x2);
    m.def("gemm_regtile4x4", &gemm_regtile4x4);
    m.def("gemm_float4", &gemm_float4);
    m.def("gemm_wmma_fp16", &gemm_wmma_fp16);
    m.def("softmax_fused", &softmax_fused);
    m.def("softmax_online", &softmax_online);
}
```

阶段2仍使用简单pybind连接完整调用链。Dispatcher、autograd、FakeTensor和`torch.compile`注册属于阶段3的生产级集成，不在此处混入性能核心实验。

---

## 14. GEMM完整参考实现 `src/gemm.cu`

下面代码强调“版本之间只有明确变化”。所有FP32手写版本共享相同输入契约、stream和错误处理。

```cpp
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <limits>


namespace {

using namespace nvcuda;


struct GemmProblem {
    int m;
    int n;
    int k;
};


GemmProblem check_base(
    const torch::Tensor& a,
    const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "a and b must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device(),
                "a and b must be on the same device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "a and b must be contiguous row-major tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0),
                "a.shape[1] must equal b.shape[0]");

    int64_t m64 = a.size(0);
    int64_t k64 = a.size(1);
    int64_t n64 = b.size(1);
    int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(m64 <= limit && n64 <= limit && k64 <= limit,
                "stage2 reference supports dimensions up to INT_MAX");

    return {
        static_cast<int>(m64),
        static_cast<int>(n64),
        static_cast<int>(k64),
    };
}


GemmProblem check_fp32(
    const torch::Tensor& a,
    const torch::Tensor& b) {
    GemmProblem p = check_base(a, b);
    TORCH_CHECK(a.scalar_type() == at::kFloat &&
                b.scalar_type() == at::kFloat,
                "this path requires float32 inputs");
    return p;
}


torch::Tensor make_fp32_output(
    const torch::Tensor& a,
    const GemmProblem& p) {
    return torch::empty({p.m, p.n}, a.options().dtype(at::kFloat));
}


bool handle_empty_or_zero_k(
    torch::Tensor& out,
    const GemmProblem& p) {
    if (p.m == 0 || p.n == 0) {
        return true;
    }
    if (p.k == 0) {
        out.zero_();
        return true;
    }
    return false;
}


__global__ void gemm_naive_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < k_size; ++k) {
            acc += a[row * k_size + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }
}


constexpr int TILE = 16;


template <int PAD>
__global__ void gemm_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k_size) {
    __shared__ float a_tile[TILE][TILE + PAD];
    __shared__ float b_tile[TILE][TILE + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int k0 = 0; k0 < k_size; k0 += TILE) {
        int a_col = k0 + tx;
        int b_row = k0 + ty;

        a_tile[ty][tx] =
            (row < m && a_col < k_size)
                ? a[row * k_size + a_col]
                : 0.0f;
        b_tile[ty][tx] =
            (b_row < k_size && col < n)
                ? b[b_row * n + col]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}


template <int RM, int RN>
__global__ void gemm_regtile_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k_size) {
    __shared__ float a_tile[TILE * RM][TILE];
    __shared__ float b_tile[TILE][TILE * RN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int base_row = blockIdx.y * (TILE * RM) + ty * RM;
    int base_col = blockIdx.x * (TILE * RN) + tx * RN;

    float acc[RM][RN] = {};

    for (int k0 = 0; k0 < k_size; k0 += TILE) {
        #pragma unroll
        for (int r = 0; r < RM; ++r) {
            int row = base_row + r;
            int col = k0 + tx;
            a_tile[ty * RM + r][tx] =
                (row < m && col < k_size)
                    ? a[row * k_size + col]
                    : 0.0f;
        }

        #pragma unroll
        for (int col_offset = 0; col_offset < RN; ++col_offset) {
            int row = k0 + ty;
            int col = base_col + col_offset;
            b_tile[ty][tx * RN + col_offset] =
                (row < k_size && col < n)
                    ? b[row * n + col]
                    : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_values[RM];
            float b_values[RN];

            #pragma unroll
            for (int r = 0; r < RM; ++r) {
                a_values[r] = a_tile[ty * RM + r][k];
            }
            #pragma unroll
            for (int col_offset = 0; col_offset < RN; ++col_offset) {
                b_values[col_offset] = b_tile[k][tx * RN + col_offset];
            }

            #pragma unroll
            for (int r = 0; r < RM; ++r) {
                #pragma unroll
                for (int col_offset = 0; col_offset < RN; ++col_offset) {
                    acc[r][col_offset] +=
                        a_values[r] * b_values[col_offset];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < RM; ++r) {
        #pragma unroll
        for (int col_offset = 0; col_offset < RN; ++col_offset) {
            int row = base_row + r;
            int col = base_col + col_offset;
            if (row < m && col < n) {
                c[row * n + col] = acc[r][col_offset];
            }
        }
    }
}


__global__ void gemm_float4_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int vector_col = blockIdx.x * blockDim.x + threadIdx.x;
    int base_col = vector_col * 4;

    if (row >= m || base_col >= n) {
        return;
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = 0; k < k_size; ++k) {
        float a_value = a[row * k_size + k];
        const float4* b4 = reinterpret_cast<const float4*>(
            b + k * n + base_col);
        float4 b_value = b4[0];
        acc.x += a_value * b_value.x;
        acc.y += a_value * b_value.y;
        acc.z += a_value * b_value.z;
        acc.w += a_value * b_value.w;
    }

    float4* c4 = reinterpret_cast<float4*>(c + row * n + base_col);
    c4[0] = acc;
}


constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_WARPS_PER_BLOCK = 4;


__global__ void gemm_wmma_fp16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k_size) {
    int warp_id = threadIdx.y;
    int tile_m = blockIdx.y * WMMA_WARPS_PER_BLOCK + warp_id;
    int tile_n = blockIdx.x;
    int row = tile_m * WMMA_M;
    int col = tile_n * WMMA_N;

    if (row >= m || col >= n) {
        return;
    }

    wmma::fragment<
        wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        half,
        wmma::row_major> a_frag;
    wmma::fragment<
        wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        half,
        wmma::row_major> b_frag;
    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k_size; k0 += WMMA_K) {
        wmma::load_matrix_sync(
            a_frag,
            a + row * k_size + k0,
            k_size);
        wmma::load_matrix_sync(
            b_frag,
            b + k0 * n + col,
            n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(
        c + row * n + col,
        c_frag,
        n,
        wmma::mem_row_major);
}


void launch_naive(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    const GemmProblem& p,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(
        (p.n + block.x - 1) / block.x,
        (p.m + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        p.m, p.n, p.k);
}


template <int PAD>
void launch_tiled(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    const GemmProblem& p,
    cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid(
        (p.n + TILE - 1) / TILE,
        (p.m + TILE - 1) / TILE);
    gemm_tiled_kernel<PAD><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        p.m, p.n, p.k);
}


template <int RM, int RN>
void launch_regtile(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    const GemmProblem& p,
    cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid(
        (p.n + TILE * RN - 1) / (TILE * RN),
        (p.m + TILE * RM - 1) / (TILE * RM));
    gemm_regtile_kernel<RM, RN><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        p.m, p.n, p.k);
}


bool aligned_for_float4(
    const torch::Tensor& b,
    const torch::Tensor& c,
    const GemmProblem& p) {
    uintptr_t b_address = reinterpret_cast<uintptr_t>(b.data_ptr());
    uintptr_t c_address = reinterpret_cast<uintptr_t>(c.data_ptr());
    return p.n % 4 == 0 &&
           b_address % alignof(float4) == 0 &&
           c_address % alignof(float4) == 0;
}


}  // namespace


torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_naive(a, b, c, p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_tiled(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_tiled<0>(a, b, c, p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_tiled_padding(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_tiled<1>(a, b, c, p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_regtile2x2(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_regtile<2, 2>(a, b, c, p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_regtile4x4(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_regtile<4, 4>(a, b, c, p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_float4(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_fp32(a, b);
    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (!aligned_for_float4(b, c, p)) {
        launch_naive(a, b, c, p, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return c;
    }

    dim3 block(16, 16);
    dim3 grid(
        ((p.n / 4) + block.x - 1) / block.x,
        (p.m + block.y - 1) / block.y);
    gemm_float4_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        p.m, p.n, p.k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


torch::Tensor gemm_wmma_fp16(torch::Tensor a, torch::Tensor b) {
    GemmProblem p = check_base(a, b);
    TORCH_CHECK(a.scalar_type() == at::kHalf &&
                b.scalar_type() == at::kHalf,
                "gemm_wmma_fp16 requires float16 inputs");

    auto c = make_fp32_output(a, p);
    if (handle_empty_or_zero_k(c, p)) {
        return c;
    }

    c10::cuda::CUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool exact =
        p.m % WMMA_M == 0 &&
        p.n % WMMA_N == 0 &&
        p.k % WMMA_K == 0;

    if (!exact) {
        // 安全但昂贵的教学fallback：显式转FP32后调用PyTorch matmul。
        // Benchmark必须把转换和供应商库调用计入端到端时间。
        return at::matmul(a.to(at::kFloat), b.to(at::kFloat));
    }

    dim3 block(32, WMMA_WARPS_PER_BLOCK);
    dim3 grid(
        p.n / WMMA_N,
        (p.m / WMMA_M + WMMA_WARPS_PER_BLOCK - 1) /
            WMMA_WARPS_PER_BLOCK);

    gemm_wmma_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        c.data_ptr<float>(),
        p.m, p.n, p.k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
```

### 14.1 必须逐段回答的问题

1. naive中同一warp对A和B的地址模式分别是什么？
2. tiled的两个barrier分别保护什么？
3. padding版本改变了哪些地址，为什么可能完全没有收益？
4. 2×2与4×4各自输出多大CTA tile？grid block数怎样变化？
5. float4为什么要求N%4、B基址和C基址同时满足条件？
6. WMMA中一个block有几个warp、输出多少行tile？
7. 不规则WMMA fallback为什么不能与“纯WMMA kernel时间”混为一谈？

### 14.2 参考代码的故意限制

- FP32 CUDA版本用于学习，不追求cuBLAS级pipeline。
- float4版本只向量化B和C，没有shared tiling；它是受控实验，不是最终最佳版本。
- WMMA只实现FP16输入，BF16由Triton路径覆盖。
- 没有实现double buffering、`cp.async`、TMA和复杂swizzle。
- `at::matmul` fallback是正确性优先的教学方案，必须报告转换成本。

这些限制不是遗漏，而是为了让每次变化仍能被新人独立解释。

---

## 15. Softmax完整参考实现 `src/softmax.cu`

本章提供两个版本：`fused`把整行暂存在shared memory中，理想情况下只从global读取一次；`online`不缓存整行，用在线状态计算统计量后第二次读取输入输出结果。

```cpp
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>


namespace {


template <typename scalar_t>
__device__ __forceinline__ float load_as_float(
    const scalar_t* pointer,
    int64_t index) {
    return static_cast<float>(pointer[index]);
}


__device__ __forceinline__ float warp_reduce_sum(float value) {
    constexpr unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}


__device__ __forceinline__ float warp_reduce_max(float value) {
    constexpr unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}


__device__ float block_reduce_sum(
    float value,
    float* warp_values) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (blockDim.x + 31) / 32;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_values[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        float first_warp_value =
            lane < num_warps ? warp_values[lane] : 0.0f;
        first_warp_value = warp_reduce_sum(first_warp_value);
        if (lane == 0) {
            warp_values[0] = first_warp_value;
        }
    }
    __syncthreads();
    return warp_values[0];
}


__device__ float block_reduce_max(
    float value,
    float* warp_values) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (blockDim.x + 31) / 32;

    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        float first_warp_value =
            lane < num_warps ? warp_values[lane] : -INFINITY;
        first_warp_value = warp_reduce_max(first_warp_value);
        if (lane == 0) {
            warp_values[0] = first_warp_value;
        }
    }
    __syncthreads();
    return warp_values[0];
}


struct OnlineState {
    float max_value;
    float normalizer;
};


__device__ __forceinline__ OnlineState combine_online(
    OnlineState left,
    OnlineState right) {
    if (left.normalizer == 0.0f) {
        return right;
    }
    if (right.normalizer == 0.0f) {
        return left;
    }

    float new_max = fmaxf(left.max_value, right.max_value);
    float new_normalizer =
        left.normalizer * expf(left.max_value - new_max) +
        right.normalizer * expf(right.max_value - new_max);
    return {new_max, new_normalizer};
}


__device__ __forceinline__ OnlineState warp_reduce_online(
    OnlineState value) {
    constexpr unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        OnlineState other{
            __shfl_down_sync(mask, value.max_value, offset),
            __shfl_down_sync(mask, value.normalizer, offset),
        };
        value = combine_online(value, other);
    }
    return value;
}


__device__ OnlineState block_reduce_online(
    OnlineState value,
    float* warp_max,
    float* warp_normalizer) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (blockDim.x + 31) / 32;

    value = warp_reduce_online(value);
    if (lane == 0) {
        warp_max[warp_id] = value.max_value;
        warp_normalizer[warp_id] = value.normalizer;
    }
    __syncthreads();

    if (warp_id == 0) {
        OnlineState first_warp_value =
            lane < num_warps
                ? OnlineState{warp_max[lane], warp_normalizer[lane]}
                : OnlineState{-INFINITY, 0.0f};
        first_warp_value = warp_reduce_online(first_warp_value);
        if (lane == 0) {
            warp_max[0] = first_warp_value.max_value;
            warp_normalizer[0] = first_warp_value.normalizer;
        }
    }
    __syncthreads();
    return {warp_max[0], warp_normalizer[0]};
}


template <typename scalar_t>
__global__ void softmax_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    extern __shared__ float row_buffer[];
    __shared__ float warp_values[32];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        row_buffer[col] =
            load_as_float(input, row * cols + col);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_buffer[col]);
    }
    float row_max = block_reduce_max(local_max, warp_values);

    float local_sum = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float numerator = expf(row_buffer[col] - row_max);
        row_buffer[col] = numerator;
        local_sum += numerator;
    }
    float row_sum = block_reduce_sum(local_sum, warp_values);

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        output[row * cols + col] =
            static_cast<scalar_t>(row_buffer[col] / row_sum);
    }
}


template <typename scalar_t>
__global__ void softmax_online_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float warp_max[32];
    __shared__ float warp_normalizer[32];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;

    OnlineState local{-INFINITY, 0.0f};
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float x = load_as_float(input, row * cols + col);
        OnlineState one{x, 1.0f};
        local = combine_online(local, one);
    }

    OnlineState total = block_reduce_online(
        local, warp_max, warp_normalizer);

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float x = load_as_float(input, row * cols + col);
        float value = expf(x - total.max_value) / total.normalizer;
        output[row * cols + col] = static_cast<scalar_t>(value);
    }
}


struct SoftmaxProblem {
    int64_t rows;
    int64_t cols;
};


SoftmaxProblem check_softmax(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kHalf ||
        x.scalar_type() == at::kBFloat16,
        "supported dtypes: float32, float16, bfloat16");
    return {x.size(0), x.size(1)};
}


int choose_threads(int64_t cols) {
    if (cols <= 128) {
        return 128;
    }
    return 256;
}


}  // namespace


torch::Tensor softmax_fused(torch::Tensor x) {
    SoftmaxProblem p = check_softmax(x);
    auto output = torch::empty_like(x);
    if (p.rows == 0 || p.cols == 0) {
        return output;
    }

    // 8192个FP32临时值需要32 KiB dynamic shared memory。
    TORCH_CHECK(p.cols <= 8192,
                "softmax_fused supports cols <= 8192; use online fallback");

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = choose_threads(p.cols);
    size_t shared_bytes = static_cast<size_t>(p.cols) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "softmax_fused",
        [&] {
            softmax_fused_kernel<scalar_t><<<
                p.rows, threads, shared_bytes, stream>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    p.rows,
                    p.cols);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor softmax_online(torch::Tensor x) {
    SoftmaxProblem p = check_softmax(x);
    auto output = torch::empty_like(x);
    if (p.rows == 0 || p.cols == 0) {
        return output;
    }

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = choose_threads(p.cols);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "softmax_online",
        [&] {
            softmax_online_kernel<scalar_t><<<
                p.rows, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    p.rows,
                    p.cols);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
```

### 15.1 两个版本的资源差异

| 版本   | Global读取 | Global写入 | 主要片上状态                    | 主要限制                  |
| :----- | :--------- | :--------- | :------------------------------ | :------------------------ |
| fused  | 理想1次x   | 1次y       | `cols × 4 bytes` dynamic shared | 行太长时shared容量        |
| online | 2次x       | 1次y       | 每thread `(m,l)` + warp部分状态 | 多一次global读取、exp开销 |

fused并不保证总是更快。短行可能受launch限制；shared占用可能降低blocks/SM；online虽然多读一次，却允许更长行且资源更稳定。

### 15.2 必须理解的identity

- max identity：负无穷。
- sum identity：0。
- online identity：`(-∞, 0)`。

`combine_online`显式处理normalizer为0的identity，避免计算`exp(-∞ - -∞)`产生NaN。

### 15.3 参考实现限制

- 每个block只处理一行；rows很少时并行度不足。
- fused把FP16/BF16输入转换为FP32 shared缓存，shared需求按4 bytes/元素计算。
- 未融合scale/mask，作为课后改造任务。
- `expf`的精度和吞吐需要与PyTorch参考和Profiler共同验证。
- 整行全为负无穷、包含正无穷或NaN时，要先定义与PyTorch一致的语义并测试。

---

## 16. Triton GEMM `python/triton_gemm.py`

安装Triton应使用目标PyTorch/GPU环境已验证的版本。下面实现不使用autotune，先让tile选择成为你能解释的受控变量。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_block * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _select_config(M, N, K):
    if M <= 32:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 4,
            "num_warps": 4,
        }
    if M <= 128:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 4,
            "num_warps": 4,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
    }


def triton_matmul(a, b, config=None):
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton path supports float16 and bfloat16")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")

    M, K = a.shape
    _, N = b.shape
    if M == 0 or N == 0:
        return torch.empty((M, N), device=a.device, dtype=torch.float32)
    if K == 0:
        return torch.zeros((M, N), device=a.device, dtype=torch.float32)

    selected = dict(_select_config(M, N, K) if config is None else config)
    num_warps = selected.pop("num_warps")
    grid = (
        triton.cdiv(M, selected["BLOCK_M"])
        * triton.cdiv(N, selected["BLOCK_N"]),
    )

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        **selected,
        num_warps=num_warps,
    )
    return c
```

### 16.1 为什么这是阶段2对照，而不是抄官方教程

你必须完成三项改造：

1. 使用本教材统一的FP32输出、shape matrix、容差和benchmark。
2. 对小M、方阵、不规则shape分别选择并解释config。
3. 固定一个shape，比较至少三组`BLOCK_M/N/K、GROUP_M、num_warps`，记录JIT后稳态结果和资源差异。

### 16.2 Triton代码怎样对应CUDA概念

| Triton概念       | CUDA/GEMM中的对应含义                      |
| :--------------- | :----------------------------------------- |
| program instance | 近似一个负责输出tile的执行单元             |
| `BLOCK_M/N/K`    | CTA/program tile和K tile                   |
| `tl.arange`      | tile内向量化索引集合                       |
| mask             | M/N/K尾部保护                              |
| `tl.dot`         | 由编译器选择并生成矩阵乘加/Tensor Core路径 |
| `num_warps`      | 一个program使用的warp数量                  |
| `GROUP_M`        | program执行顺序调整，促进L2复用            |

### 16.3 必做Config实验

```python
CONFIGS = [
    dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, GROUP_M=4, num_warps=4),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, GROUP_M=4, num_warps=4),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8),
]
```

对`M=1/32/128/1024`分别比较。不能只保留每个shape最优config，还要解释失败config为什么浪费tile、降低并行度或增加资源。

---

## 17. 正确性测试

### 17.1 `tests/test_gemm.py`

```python
import pathlib
import sys

import pytest
import torch

import stage2_kernel_lab as ops

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "python"))
from triton_gemm import triton_matmul


FP32_SHAPES = [
    (0, 7, 5),
    (7, 0, 5),
    (7, 9, 0),
    (1, 1, 1),
    (15, 17, 16),
    (16, 16, 16),
    (17, 15, 19),
    (123, 145, 67),
    (255, 257, 511),
    (512, 512, 512),
]

FP32_IMPLS = [
    ops.gemm_naive,
    ops.gemm_tiled,
    ops.gemm_tiled_padding,
    ops.gemm_regtile2x2,
    ops.gemm_regtile4x4,
    ops.gemm_float4,
]


@pytest.fixture(autouse=True)
def disable_tf32_for_reference():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.parametrize("shape", FP32_SHAPES)
@pytest.mark.parametrize("fn", FP32_IMPLS)
def test_fp32_gemm(fn, shape):
    M, N, K = shape
    torch.manual_seed(M * 1000000 + N * 1000 + K)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    actual = fn(a, b)
    expected = a @ b
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("shape", [
    (16, 16, 16),
    (64, 64, 64),
    (128, 256, 512),
    (123, 145, 67),
])
def test_wmma_fp16_and_fallback(shape):
    M, N, K = shape
    torch.manual_seed(7)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    actual = ops.gemm_wmma_fp16(a, b)
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (1, 4096, 4096),
    (32, 4096, 4096),
    (123, 145, 67),
    (512, 512, 512),
])
def test_triton_gemm(dtype, shape):
    M, N, K = shape
    torch.manual_seed(11)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    actual = triton_matmul(a, b)
    expected = a.float() @ b.float()
    tolerance = 3e-2 if dtype == torch.float16 else 8e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance)


def test_noncontiguous_rejected():
    a = torch.randn(17, 19, device="cuda").t()
    b = torch.randn(17, 23, device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.gemm_tiled(a, b)


def test_dtype_mismatch_rejected():
    a = torch.randn(8, 8, device="cuda", dtype=torch.float32)
    b = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float32"):
        ops.gemm_tiled(a, b)


def test_float4_unaligned_contiguous_fallback():
    M, N, K = 17, 64, 19
    a = torch.randn(M, K, device="cuda")
    storage = torch.randn(K * N + 1, device="cuda")
    b = storage[1:].view(K, N)
    assert b.is_contiguous()
    actual = ops.gemm_float4(a, b)
    torch.testing.assert_close(actual, a @ b, rtol=2e-3, atol=2e-3)


def test_gemm_current_stream():
    stream = torch.cuda.Stream()
    a = torch.randn(257, 129, device="cuda")
    b = torch.randn(129, 255, device="cuda")
    with torch.cuda.stream(stream):
        actual = ops.gemm_regtile2x2(a, b)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    torch.testing.assert_close(actual, a @ b, rtol=2e-3, atol=2e-3)
```

#### 为什么关闭TF32

手写FP32 SIMT kernel使用FP32乘加，而PyTorch在某些GPU/版本可能允许FP32输入走TF32 Tensor Core。正确性reference若不固定该设置，误差来源会混合“实现错误”和“数学路径不同”。性能实验可以另开TF32基线，但必须单独命名并记录。

#### 误差报告不能只有assert

为每个dtype/shape额外保存：max absolute、max relative、mean absolute、P99 absolute、NaN/Inf数量。测试阈值只负责拦截错误，报告负责解释误差。

### 17.2 `tests/test_softmax.py`

```python
import pytest
import torch

import stage2_kernel_lab as ops


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SHAPES = [
    (0, 17),
    (7, 0),
    (1, 1),
    (1, 31),
    (2, 32),
    (17, 33),
    (32, 127),
    (32, 128),
    (32, 129),
    (1024, 1024),
    (17, 4097),
]


def tolerance(dtype):
    if dtype == torch.float32:
        return 2e-5
    if dtype == torch.float16:
        return 3e-3
    return 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn", [ops.softmax_fused, ops.softmax_online])
def test_softmax(dtype, shape, fn):
    rows, cols = shape
    torch.manual_seed(rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype) * 10
    actual = fn(x)
    expected = torch.softmax(x, dim=-1)
    tol = tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", [ops.softmax_fused, ops.softmax_online])
def test_softmax_properties(dtype, fn):
    x = torch.randn(127, 257, device="cuda", dtype=dtype)
    y = fn(x)
    tol = tolerance(dtype)
    torch.testing.assert_close(
        y.float().sum(dim=-1),
        torch.ones(127, device="cuda"),
        rtol=tol,
        atol=tol,
    )
    assert torch.all(y >= 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_equal(dtype):
    x = torch.full((17, 33), 100.0, device="cuda", dtype=dtype)
    expected = torch.full_like(x, 1.0 / 33)
    for fn in (ops.softmax_fused, ops.softmax_online):
        actual = fn(x)
        tol = tolerance(dtype)
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def test_fused_shared_limit_and_online_fallback():
    x = torch.randn(2, 9000, device="cuda")
    with pytest.raises(RuntimeError, match="8192"):
        ops.softmax_fused(x)
    actual = ops.softmax_online(x)
    torch.testing.assert_close(
        actual, torch.softmax(x, dim=-1), rtol=2e-5, atol=2e-5)


def test_noncontiguous_rejected():
    x = torch.randn(17, 33, device="cuda").t()
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.softmax_online(x)


def test_softmax_current_stream():
    stream = torch.cuda.Stream()
    x = torch.randn(127, 257, device="cuda")
    with torch.cuda.stream(stream):
        actual = ops.softmax_online(x)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    torch.testing.assert_close(
        actual, torch.softmax(x, dim=-1), rtol=2e-5, atol=2e-5)
```

需要自行补充：NaN、正无穷、全负无穷、scale、causal mask、全mask行和固定回归shape。先观察PyTorch语义，再把接口契约写进README。

### 17.3 构建与运行

```bash
python -m pip install -v -e .
pytest -q tests/test_gemm.py
pytest -q tests/test_softmax.py
```

测试失败时禁止先跑Profiler。先把失败缩小到最小shape，确认索引、尾部、layout、dtype和累加协议。

---

## 18. Benchmark与Profiler工程

### 18.1 `benchmark/bench_gemm.py`

```python
import csv
import pathlib
import statistics
import sys

import torch

import stage2_kernel_lab as ops

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "python"))
from triton_gemm import triton_matmul


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def cuda_bench(fn, warmup=20, repeats=40, inner=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)

    return {
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def tflops(M, N, K, milliseconds):
    return 2.0 * M * N * K / (milliseconds * 1e-3) / 1e12


def environment():
    index = torch.cuda.current_device()
    return {
        "gpu": torch.cuda.get_device_name(index),
        "capability": str(torch.cuda.get_device_capability(index)),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "tf32": str(torch.backends.cuda.matmul.allow_tf32),
    }


def fp32_implementations(a, b):
    return {
        "torch_fp32": lambda: a @ b,
        "naive": lambda: ops.gemm_naive(a, b),
        "tiled": lambda: ops.gemm_tiled(a, b),
        "tiled_padding": lambda: ops.gemm_tiled_padding(a, b),
        "regtile2x2": lambda: ops.gemm_regtile2x2(a, b),
        "regtile4x4": lambda: ops.gemm_regtile4x4(a, b),
        "float4": lambda: ops.gemm_float4(a, b),
    }


def low_precision_implementations(a, b):
    result = {
        # native输出为低精度；用于供应商库性能参考，契约差异必须记录。
        "torch_native": lambda: a @ b,
        # 输入先转FP32后做FP32 GEMM；输出契约一致，但计算路径不同。
        "torch_fp32_contract": lambda: a.float() @ b.float(),
        "triton_fp32_output": lambda: triton_matmul(a, b),
    }
    if a.dtype == torch.float16:
        result["wmma_fp32_output"] = lambda: ops.gemm_wmma_fp16(a, b)
    return result


def main():
    torch.manual_seed(7)
    print(environment())

    shapes = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (1, 4096, 4096),
        (128, 4096, 4096),
        (4096, 11008, 4096),
        (123, 145, 67),
        (255, 257, 511),
    ]

    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for M, N, K in shapes:
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)

            implementations = (
                fp32_implementations(a, b)
                if dtype == torch.float32
                else low_precision_implementations(a, b)
            )

            for name, fn in implementations.items():
                # naive对大问题会极慢，不让教学基线拖垮整次运行。
                if name == "naive" and M * N * K > 512**3:
                    continue
                try:
                    stats = cuda_bench(fn)
                except RuntimeError as error:
                    print("SKIP", dtype, (M, N, K), name, error)
                    continue

                row = {
                    "dtype": str(dtype),
                    "M": M,
                    "N": N,
                    "K": K,
                    "implementation": name,
                    **stats,
                    "effective_tflops": tflops(
                        M, N, K, stats["median_ms"]),
                }
                rows.append(row)
                print(row)

    output = pathlib.Path("reports/version_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

大shape可能超过显存或运行过久，允许按GPU容量缩小，但不要删除小M、长条、不规则这些类别。

### 18.2 低精度基线为什么有两个

`torch_native`最接近实际供应商低精度GEMM性能，但输出通常是低精度；教材WMMA/Triton输出FP32。`torch_fp32_contract`输出一致，却包含转换并使用不同计算路径。

因此报告中：

- 不能把任一基线称为完全同协议而不说明差异。
- 需要分别报告kernel-only和端到端转换成本。
- 可以把native库当性能上限参考，把FP32 contract当正确性/接口参考。

### 18.3 Triton JIT时间

第一次调用Triton包含JIT编译，稳态benchmark的warmup会把它排除。必须单独记录首次调用：

```python
import time

torch.cuda.synchronize()
start = time.perf_counter()
triton_matmul(a, b)
torch.cuda.synchronize()
first_call_ms = (time.perf_counter() - start) * 1000
print("Triton first call including JIT:", first_call_ms, "ms")
```

不能拿首次Triton时间与已编译CUDA kernel比较，也不能在部署讨论中完全忽略JIT成本。

### 18.4 `benchmark/bench_softmax.py`

```python
import pathlib

import torch

import stage2_kernel_lab as ops
from bench_gemm import cuda_bench


def effective_gbps(x, milliseconds):
    minimum_bytes = 2 * x.numel() * x.element_size()
    return minimum_bytes / (milliseconds * 1e-3) / 1e9


def main():
    torch.manual_seed(13)
    results = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for rows, cols in [
            (1, 32),
            (1, 4096),
            (32, 128),
            (32, 4097),
            (1024, 128),
            (1024, 1024),
            (4096, 4096),
        ]:
            x = torch.randn(rows, cols, device="cuda", dtype=dtype)
            implementations = {
                "torch": lambda: torch.softmax(x, dim=-1),
                "fused": lambda: ops.softmax_fused(x),
                "online": lambda: ops.softmax_online(x),
            }
            for name, fn in implementations.items():
                stats = cuda_bench(fn)
                record = {
                    "dtype": str(dtype),
                    "rows": rows,
                    "cols": cols,
                    "implementation": name,
                    **stats,
                    "effective_gbps": effective_gbps(
                        x, stats["median_ms"]),
                }
                results.append(record)
                print(record)

    output = pathlib.Path("reports/softmax_results.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(map(str, results)) + "\n")


if __name__ == "__main__":
    main()
```

有效GB/s只按最少“一读一写”计算。fused、online和PyTorch实际DRAM流量不同，应在NCU中另看绝对bytes。

### 18.5 `benchmark/profile_entry.py`

```python
import argparse

import torch

import stage2_kernel_lab as ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(17)
    if args.op.startswith("gemm"):
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        fn = getattr(ops, args.op)
        for _ in range(args.iters):
            fn(a, b)
    elif args.op.startswith("softmax"):
        x = torch.randn(1024, 4096, device="cuda")
        fn = getattr(ops, args.op)
        for _ in range(args.iters):
            fn(x)
    else:
        raise ValueError(f"unknown op: {args.op}")

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
```

### 18.6 NSYS命令

```bash
mkdir -p reports/nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o reports/nsys/gemm_benchmark \
  python benchmark/bench_gemm.py
```

按时间线检查：首次初始化/JIT、CPU gap、dtype转换、分配、launch间隔、同步、Kernel持续时间和排列。先确认目标Kernel真的占主要时间，再进入NCU。

### 18.7 NCU命令

```bash
mkdir -p reports/ncu
ncu \
  --target-processes all \
  --set full \
  --force-overwrite \
  -o reports/ncu/gemm_regtile2x2 \
  python benchmark/profile_entry.py \
  --op gemm_regtile2x2 \
  --iters 1
```

Profiler权限不足时按服务器规范处理，不要在教材脚本中默认写`sudo`。`--set full`可能运行较慢；先用较轻section确定方向，再针对一个Kernel收集完整报告。

### 18.8 PTX/SASS与编译报告

找到扩展共享库：

```bash
find . -name 'stage2_kernel_lab*.so'
```

查看SASS：

```bash
cuobjdump --dump-sass path/to/stage2_kernel_lab.so > reports/stage2.sass
```

搜索方向：

- WMMA/MMA/HMMA相关矩阵乘加指令。
- vectorized版本是否出现宽load/store。
- local memory load/store是否支持spill假设。
- FP32版本的FMA指令路径。

指令名称随GPU代际变化。结论写“在目标GPU/工具链观察到什么”，不要把某一代SASS名字当成跨代永久规则。

---

## 19. 八轮执行安排

“一轮”通常是一周，但验收优先于日历。每轮只产出一个可复现结论。

### 第一轮：冻结契约和基线

- 建工程、环境快照、shape matrix和测试骨架。
- 完成naive，手算FLOPs/bytes和地址模式。
- 建PyTorch/cuBLAS与CUDA Event基线。

**出口：** naive全shape正确，性能数字可复现，能解释它为什么慢。

### 第二轮：Shared tiling

- 独立实现tiled，不复制参考核心代码。
- 推导global读取减少、shared占用和barrier。
- naive/tiled受控对照与第一次NCU。

**出口：** 指标支持或修正你的复用模型。

### 第三轮：Register tile 2×2

- 独立实现2×2。
- 推导shared读取、CTA tile、grid和register变化。
- 方阵、小M、不规则shape对照。

**出口：** 能说明2×2在哪些shape获益，在哪里无效。

### 第四轮：4×4退化与竞争假设

- 实现4×4，保留失败数据。
- 提出至少两个竞争假设。
- 用编译报告、grid/wave、NCU逐个排除。

**出口：** 一份“优化变慢”的完整复盘，而不是删除慢版本。

### 第五轮：Padding、float4和fallback

- padding先算bank再运行。
- float4验证alignment、指令、请求和Duration。
- 覆盖N非4整数倍和不对齐输入。

**出口：** 至少一个“经验技巧有效或无效”的因果结论。

### 第六轮：WMMA与Triton

- FP16 WMMA精确shape与fallback。
- FP16/BF16 Triton，修改三组config。
- 用指令/NCU确认Tensor Core路径。
- 与供应商库诚实比较契约差异。

**出口：** 低精度主案例、指令证据和主流栈对照。

### 第七轮：Softmax迁移

- fused与online正确性、读写推导和shape sweep。
- NCU分析短行/长行、少rows/多rows。
- 选择一次scale或mask融合改造。

**出口：** 第二算子达到L3～L4，不只是GEMM经验复述。

### 第八轮：报告、回归与延迟复测

- 汇总版本总表、主报告和失败案例。
- 固定性能回归shape和阈值。
- 清理README、一键测试/benchmark/profile命令。
- 一周后关闭教材重写naive、tiled、regtile2x2。

**出口：** 陌生工程师能复现数字，你能脱离文档解释全部因果。

---

## 20. 性能报告模板

每一个主要版本使用同一模板：

```text
# 版本名称

## 1. 问题和契约
shape、dtype、layout、累加、输出、边界、stream

## 2. Baseline
naive、PyTorch/供应商库、前一版本

## 3. 变化
只描述本版本相对前一版本改变的主要变量

## 4. 运行前预测
FLOPs、bytes、load/store、tile、register、shared、grid/wave、指令

## 5. 正确性
shape matrix、误差分布、极值、失败/回归

## 6. 测量协议
GPU、软件、编译、seed、warmup、repeats、计时器、统计

## 7. 结果
median/P90/P95、TFLOP/s或GB/s、与各baseline比例

## 8. Profiler证据
NSYS层级；NCU八步；绝对bytes/requests/sectors；资源；stall；指令

## 9. 竞争假设和排除
至少两个假设、最小实验、支持/反驳证据

## 10. 结论
为什么变快/慢、代价、适用shape、不可推广范围

## 11. 下一步
只列一个最高价值实验
```

### 20.1 性能回归最小表

| 字段            | 含义                                   |
| :-------------- | :------------------------------------- |
| environment_id  | GPU、驱动、CUDA、PyTorch、编译参数哈希 |
| implementation  | 精确版本名                             |
| shape/dtype     | M/N/K或rows/cols与dtype                |
| contract        | 累加、输出、是否含转换                 |
| median/P95      | 稳态统计                               |
| correctness     | 最大/P99误差与测试版本                 |
| baseline_commit | 基线代码版本                           |
| change          | 相对基线百分比                         |
| status          | pass、investigate、expected-change     |

回归阈值必须结合噪声设定。例如历史波动P95为3%，就不能用1%作为通用失败阈值。性能退化先复测环境，再判断代码。

---

## 21. 阶段2原表格逐项覆盖

| 原技术问题             | 必须达到的深度                                    | 教材位置          | 验收证据                              |
| :--------------------- | :------------------------------------------------ | :---------------- | :------------------------------------ |
| 输入和正确性协议       | FP32/FP16/BF16、真实/边界shape、累加与容差        | 模块1、17章       | shape matrix、误差报告、错误/回归测试 |
| Naive访存模型          | load/FMA、FLOPs/bytes、算术强度                   | 模块2、14章       | 手写推导与NCU流量方向                 |
| Shared tiling          | CTA tile、复用倍数、同步、shared和bank            | 模块3、14章       | naive/tiled受控对照                   |
| Register tiling        | thread tile、shared复用、register与occupancy      | 模块4、14章       | 2×2/4×4编译和NCU对照                  |
| Benchmark协议          | 固定环境、warmup、统计、baseline                  | 模块1、18章       | 一键benchmark与环境快照               |
| Padding与bank conflict | 地址映射、conflict绝对量和Duration                | 模块5、14章       | padding有效或无效的因果结论           |
| Vectorized access      | alignment、指令、transaction、尾部和fallback      | 模块5、14章       | N整除/非整除、对齐/非对齐对照         |
| Register tile退化      | register、blocks/SM、eligible warp、stall和shape  | 模块4、19章       | 优化变慢的竞争假设复盘                |
| Tensor Core            | FP16/BF16、FP32累加、fragment/layout、shape和指令 | 模块6、14/16章    | WMMA/Triton、fallback和MMA指令证据    |
| 性能上限               | TFLOP/s、带宽、理论峰值、供应商库比例和适用范围   | 模块1/7、18章     | 多shape版本结果表                     |
| Profiler因果链         | NSYS分层、NCU八步、绝对量、stall和Roofline        | 模块7、18/20章    | 完整主报告                            |
| Triton或CUTLASS对照    | 修改config、同协议比较、开发效率与控制粒度        | 模块8、16章       | 非教程式Triton实现                    |
| 第二算子迁移           | Softmax算法、读写、映射、精度、融合和Profiler     | 模块9、15/17/18章 | Softmax案例与一次NCU分析              |

### 21.1 KER出口

- [ ] KER-01：dtype、shape、stride、device、alignment和空输入契约完整。
- [ ] KER-02：M/N/K尾部、非tile整数倍、小shape和奇数维度正确。
- [ ] KER-03：累加dtype、容差、极值和误差分布明确。
- [ ] KER-04：当前device/stream正确，无隐藏同步。
- [ ] KER-05：向量路径不满足alignment时安全fallback。
- [ ] KER-06：真实、边界、不规则、压力shape matrix完整。
- [ ] KER-07：naive、PyTorch/供应商库和当前最佳基线同时保留。
- [ ] KER-08：根据shape/dtype选择实现，不用一个配置覆盖所有输入。
- [ ] KER-09：至少一条可修改config的非教程式Triton实现。
- [ ] KER-10：API、测试、benchmark、profiling和文档闭环。

### 21.2 GEMM出口

- [ ] GEMM-01：能计算naive每输出load/FMA和算术强度。
- [ ] GEMM-02：能推导shared tiling的global读取减少和同步点。
- [ ] GEMM-03：能解释2×2/4×4如何改变shared读取和register。
- [ ] GEMM-04：padding先算bank再实验，不套经验。
- [ ] GEMM-05：vectorization有alignment、指令和尾部证据。
- [ ] GEMM-06：FP16 WMMA和FP16/BF16 Triton使用FP32累加。
- [ ] GEMM-07：能画CTA、warp/program、thread和MMA tile映射。
- [ ] GEMM-08：能解释double buffering的load/compute overlap和资源代价；不要求本阶段手写。
- [ ] GEMM-09：能计算有效TFLOP/s并诚实解释与cuBLAS差距。
- [ ] GEMM-10：能分析小M、长条和不规则shape退化。

### 21.3 PERF出口

- [ ] PERF-01～04：测量协议、统计、受控变量和计时器全部固定。
- [ ] PERF-05：NSYS能识别CPU gap、launch、同步、转换和拷贝。
- [ ] PERF-06：NCU按八步问题树分析。
- [ ] PERF-07：吞吐百分比与绝对工作量分开。
- [ ] PERF-08：stall与eligible warp、资源、指令和源码联动。
- [ ] PERF-09：cache/shared/DRAM同时看bytes、requests、sectors和Duration。
- [ ] PERF-10：Roofline的FLOPs、bytes、dtype和层级定义正确。
- [ ] PERF-11：用编译报告/PTX/SASS验证vector、MMA或spill中的至少一项。
- [ ] PERF-12：固定shape、阈值、基线和历史结果建立性能回归。
- [ ] PERF-13：至少保留一次反直觉或优化变慢案例。
- [ ] PERF-14：报告完整回答变化、原因、证据、代价和边界。
- [ ] PERF-15：能把相对吞吐提升换算为相同工作量下的GPU时间节省；服务成本留到入行后。

### 21.4 Softmax出口

- [ ] OPS-01：能推导组合式、片上融合和online算法的读写/状态。
- [ ] OPS-02：能根据长短行选择warp/block映射。
- [ ] OPS-03：能证明瓶颈是bandwidth、latency、launch或并行度中的哪一类。
- [ ] OPS-04：能说明scale/mask融合的收益、语义和限制。

阶段2选择Softmax作为第二算子时，不要求同时完成Norm、RoPE和Attention实现；它们留作后续迁移题。

---

## 22. 新人正式进入阶段3前的闭卷口试

每个回答必须能指向自己的代码、推导或Profiler报告。

1. `C=A×B`中M/N/K分别怎样进入地址和FLOPs？
2. naive每个输出读取多少A/B，为什么真实DRAM bytes可能少于模型？
3. TILE16怎样把代码级global读取理论减少16倍？
4. tiled kernel两个barrier分别保护什么？
5. 对比当前`gemm_tiled`与`gemm_tiled_padding`的A/B Shared Memory访问映射，padding为什么可能无效甚至制造bank conflict？怎样用lane地址和NCU证明？
6. 2×2 thread tile怎样减少shared读取？代价是什么，怎样验证收益大于代价？
7. 4×4慢于2×2时，至少有哪些竞争假设？
8. occupancy下降为什么不自动等于性能下降？
9. float4改变了bytes、request、sector还是指令？怎样验证？
10. M/N/K非tile整数倍分别怎样处理？
11. CTA tile、warp tile、thread tile和MMA tile是什么关系？
12. FP16、BF16、TF32的存储、范围、精度和累加有什么不同？
13. 怎样证明实际用了Tensor Core，而不是只看源码？
14. 当前教学WMMA实现在不规则shape上为什么选择fallback？还可采用哪些正确处理方案，fallback成本怎样报告？
15. Triton `BLOCK_M/N/K、GROUP_M、num_warps`分别控制什么？
16. 为什么Triton首次调用与稳态时间必须分开？
17. NSYS和NCU分别先回答什么问题？
18. NCU八步问题树的顺序是什么？每一步分别要排除哪类错误结论？
19. DRAM throughput百分比上升为什么仍可能变慢？
20. stall分析为什么要先看eligible warp和issue slot？
21. Roofline的横轴、纵轴和ridge point是什么？
22. 有效TFLOP/s与实际执行指令数有什么区别？
23. Online Softmax `(m,l)`怎样合并，为什么数值稳定？
24. 本文缓存整行的`softmax_fused`为什么可能慢于维护`(m,l)`状态的`softmax_online`？
25. 你的最佳GEMM版本在哪些shape不适用，为什么？

能够背出定义但不能给出本项目数字、shape和证据，仍未通过。

---

## 23. 课后题与参考答案

### 23.1 `1024³` Naive GEMM手算

**题目：** `M=N=K=1024`，block=`16×16`。

**答案：**

```text
grid = (ceil(1024/16), ceil(1024/16)) = (64,64)
blocks = 4096
threads/block = 256
总threads = 1,048,576
warps/block = 8
总warps = 32,768

每thread：1024 FMA = 2048 FLOPs
总有效FLOPs = 2 × 1024³ = 2,147,483,648 ≈ 2.147 GFLOPs

naive代码级每输出流量 = 8×1024 + 4 = 8196 bytes
全部输出约8.594 GB代码级读取/写入机会

理想最低DRAM流量
= 4 × (1024² + 1024² + 1024²)
= 12,582,912 bytes ≈ 12.58 MB
```

8.594 GB不是预言的实测DRAM bytes；cache会复用。它展示naive代码与理想复用之间的空间。

### 23.2 TILE16读取复用

**题目：** 一个16×16输出tile、一个K=16分块，naive和tiled各产生多少A/B元素load？

**答案：**

```text
naive：256输出 × 16A + 256输出 × 16B = 8192元素
tiled：A tile 256 + B tile 256 = 512元素
代码级减少 = 8192/512 = 16倍
```

### 23.3 Register tile输出范围

**题目：** block固定16×16 threads。2×2和4×4每block输出多大C tile？

**答案：**

```text
2×2：每thread 2×2，CTA输出(16×2) × (16×2) = 32×32
4×4：每thread 4×4，CTA输出64×64
```

对同一M/N，2×2的block数约为普通16×16 tiled的1/4；4×4约为1/16。复用增加，但grid并行度和尾部利用率可能下降。

### 23.4 不规则tile利用率

**题目：** `M=123,N=145,K=67`，`TM=TN=32,TK=16`。

**答案：**

```text
M覆盖 = ceil(123/32)×32 = 128
N覆盖 = ceil(145/32)×32 = 160
输出tile利用率 = 123×145 / (128×160) ≈ 87.1%

K覆盖 = ceil(67/16)×16 = 80
K利用率 = 67/80 = 83.75%

三维粗略有效比例 ≈ 87.1% × 83.75% ≈ 72.9%
```

这个比例不等于性能比例，但能预测无效load/FMA和边界分支方向。

### 23.5 当前tiled映射的Padding

**题目：** 为什么`a_tile[16][17]`、`b_tile[16][17]`可能不比无padding版本快？

**答案：** 当前访问中，warp对A常表现为同一行同一k的广播组，对B常表现为相邻tx或相同地址组，并不一定存在leading-dimension stride造成的严重bank conflict。padding增加shared占用，却可能没有减少conflict。必须以真实warp线性排列、访问地址和NCU conflict绝对量为准。

### 23.6 Float4

**题目：** `float4`是否把最低数学流量减少4倍？

**答案：** 否。仍需读取相同A/B元素并写相同C元素。它可能把4条scalar load/store合成更宽指令，改变指令数和请求组织；sector/DRAM bytes未必减少。

### 23.7 WMMA映射

**题目：** 参考实现block=`dim3(32,4)`，每warp负责`16×16`输出。一个block输出什么范围？

**答案：** 4个warp沿M方向各负责一个16×16 tile，因此一个block覆盖`64×16`输出区域。grid.x=`N/16`，grid.y=`ceil((M/16)/4)`；每个warp沿K执行`K/16`次MMA。

### 23.8 理想GEMM算术强度

**答案：** 对FP32理想每个A/B只读一次、C写一次：

```text
AI_ideal = 2MNK / [4(MK + KN + MN)]
```

方阵M=N=K=L且L很大时约为`L/6 FLOPs/byte`，随矩阵增大而提高；naive代码级模型却接近0.25，说明数据复用对GEMM至关重要。

### 23.9 Online Softmax合并证明

两段分别有：

```text
l1 = Σ(x属于段1) exp(x-m1)
l2 = Σ(x属于段2) exp(x-m2)
m = max(m1,m2)
```

把两段都改写到同一个m：

```text
Σ段1 exp(x-m) = exp(m1-m) × l1
Σ段2 exp(x-m) = exp(m2-m) × l2
```

相加得到：

```text
l = l1×exp(m1-m) + l2×exp(m2-m)
```

所有指数的自变量不大于0，避免对大正数直接exp。

### 23.10 Throughput反直觉

**题目：** DRAM throughput从40%升到60%，Duration却变长，是否说明显存更忙所以更快？

**答案：** 不能。可能是实际DRAM bytes增加，导致带宽利用率更高但工作更多；也可能其他瓶颈延长时间。必须同时比较绝对bytes、requests、sectors、有效FLOPs和Duration。

---

## 24. 最小术语表

| 术语                   | 阶段2中的准确含义                                   |
| :--------------------- | :-------------------------------------------------- |
| GEMM                   | 通用矩阵乘，`C=A×B`                                 |
| CTA tile               | 一个thread block负责的输出区域                      |
| Warp tile              | CTA tile中由一个warp负责的区域                      |
| Thread tile            | 一个thread在寄存器中累加的输出区域                  |
| K tile                 | 每轮加载并参与乘加的K维分块                         |
| Register tiling        | 每thread计算多个输出以复用shared数据                |
| WMMA                   | CUDA C++的warp级矩阵乘加API                         |
| MMA/HMMA               | 低层矩阵乘加指令族的泛称，具体名字依GPU代际         |
| Fragment               | 一个warp协作持有的WMMA逻辑矩阵片段                  |
| Leading dimension      | 相邻矩阵行/列起点在内存中的跨度                     |
| Epilogue               | GEMM累加后写回前的缩放、bias、activation等阶段      |
| Padding                | 改变片上数组leading dimension以调整bank映射         |
| Double buffering       | 两套片上缓冲交替加载/计算以重叠数据搬运             |
| Pipeline stage         | 软件流水中一轮load/compute所处阶段                  |
| Effective TFLOP/s      | 按数学`2MNK`除以时间得到的有效吞吐                  |
| Effective GB/s         | 按算法最低必要bytes除以时间得到的有效带宽           |
| Roofline               | 用算术强度连接计算峰值与带宽峰值的上限模型          |
| Ridge point            | 计算roof与带宽roof相交的算术强度                    |
| Eligible warp          | 当前没有stall、可以被scheduler发射的warp            |
| Warp stall             | warp因依赖、数据、同步或执行管线暂时不能发射        |
| Wave tail              | 最后一波block不足以占满全部SM                       |
| Shape sensitivity      | 同一实现性能随M/N/K结构变化的现象                   |
| Autotune               | 对给定shape/dtype试验多个配置并选择结果             |
| JIT                    | 运行时根据代码和参数编译                            |
| Triton program         | Triton中负责一个数据tile的程序实例                  |
| CUTLASS                | NVIDIA高性能线性代数CUDA C++模板库                  |
| CuTe                   | CUTLASS中的layout、copy和MMA组合抽象                |
| Online Softmax         | 用可合并 `(m,l)` 状态流式计算稳定Softmax统计量      |
| Fused operator         | 在一个Kernel中完成多个数学步骤以减少中间读写/launch |
| Performance regression | 固定协议下相对历史基线的性能退化                    |

---

## 25. 阶段2明确不解决什么

- 不要求手写GEMM击败cuBLAS。
- 不同时深入Triton、CUTLASS、CuTe、PTX和SASS五条主线；主流栈只选Triton落地。
- 不学习Hopper/Blackwell专属TMA、WGMMA、cluster和persistent kernel细节。
- 不把`--use_fast_math`作为免费性能开关；需要单独精度实验。
- 不只测方阵和整齐shape。
- 不删除慢版本、失败shape和反直觉数据。
- 不把occupancy、带宽百分比、Tensor Core利用率中的任一项当作最终结论。
- 不在正确性、测量协议和baseline未固定前开始调参。
- 不把Triton首次JIT时间与稳态Kernel时间混为一谈。
- 不在阶段2扩展到完整Attention、FlashAttention、vLLM或分布式推理。

**阶段2的工作边界：完成可复现的GEMM旗舰项目、Reduction类项目和非Reduction迁移项目，并能用数据证明每次变化为什么有效、无效或退化。**

---

## 26. 扩展教材一：LayerNorm与RMSNorm

### 26.1 从公式到工作量

```text
LayerNorm:
mean = Σx/H
var  = Σ(x-mean)²/H
y    = (x-mean)/sqrt(var+eps) × gamma + beta

RMSNorm:
rms  = sqrt(Σx²/H + eps)
y    = x/rms × gamma
```

直观上，LayerNorm同时“平移到均值0”和“缩放到方差1”；RMSNorm只根据平方均值缩放。朴素组合实现会多次读写整行，融合Kernel把统计量、归一化和仿射变换放在一次launch中。

### 26.2 映射例题

输入`[tokens,H]`，每行独立。`H≤1024`时可用一个block处理一行：每线程处理多个元素，在FP32中累加局部统计量，再做warp/block reduction。`H`很大时shared/register和单行并行效率需要重新权衡；tokens很少时grid并行度不足。

### 26.3 稳定统计

`E[x²]-E[x]²`在两个大而接近的数相减时可能取消。Welford状态`(count,mean,M2)`可合并：

```text
delta = mean_b - mean_a
count = count_a + count_b
mean  = mean_a + delta × count_b/count
M2    = M2_a + M2_b + delta² × count_a×count_b/count
```

这增加状态和指令，但提高稳定性。实验必须同时比较误差、register和Duration。

### 26.4 必做实验

比较PyTorch组合、融合RMSNorm、融合LayerNorm/Welford；测试`H=31/32/33/127/128/129/1024/4097`、FP32/FP16/BF16、全相等和大偏置小方差输入。报告最低bytes、实际DRAM bytes、误差分布和Duration。

---

## 27. 扩展教材二：RoPE、SwiGLU与Epilogue融合

### 27.1 RoPE

对一对分量`(x0,x1)`：

```text
y0 = x0 cosθ - x1 sinθ
y1 = x0 sinθ + x1 cosθ
```

`θ`由position和频率决定。实现前先确定布局：成对相邻还是前后半区配对；索引必须包含batch/token/head/pair。Prefill有很多token，Decode常只有一个新token，二者并行度不同。

### 27.2 SwiGLU

```text
SwiGLU(a,b) = SiLU(a) × b
SiLU(a) = a × sigmoid(a)
```

若`a/b`来自两个投影输出，融合可避免中间activation写回。收益来自减少global读写与launch；代价是特殊函数吞吐、寄存器、布局组合和尾部。

### 27.3 GEMM Epilogue

Accumulator写回前可融合：

```text
out = activation(alpha × accumulator + bias + residual)
```

必须比较完整子图，而非只比较GEMM主循环。低精度输出还要定义rounding、saturation、scale和vector store alignment。

### 27.4 必做实验

选RoPE或SwiGLU：先跑PyTorch组合基线，再写融合Kernel；覆盖真实H/D、奇数尾部、未对齐输入和非默认stream。手算融合前后读写，使用NSYS确认launch减少，使用NCU确认bytes方向。

---

## 28. 扩展教材三：Attention IO与KV Cache

### 28.1 Shape

```text
Q: [B, Nh, Sq, D]
K,V: [B, Nkv, Skv, D]
Scores: [B, Nh, Sq, Skv]
O: [B, Nh, Sq, D]
```

MHA有`Nh=Nkv`；GQA让多个Q head共享一个KV head；MQA常只有一个KV head。KV容量近似：

```text
2 × layers × tokens × Nkv × D × bytes_per_element
```

例：32层、Nkv=8、D=128、FP16、4096 token：约`2×32×4096×8×128×2 ≈ 536.9 MB`。

### 28.2 Prefill与Decode

- Prefill：`Sq`较大，QK/PV矩阵较大，计算和并行度高；完整scores为`Sq×Skv`。
- Decode：常见`Sq=1`，每步读取历史KV，算术强度低，容易受带宽和并行度限制。

### 28.3 FlashAttention的IO推导

传统路径把scores和probability写到HBM再读回。分块算法把Q/K/V tile放到片上，使用Online Softmax状态合并，不完整materialize`Sq×Skv`中间矩阵。它可能用recompute增加少量FLOPs换取大量HBM IO减少。

### 28.4 连续与分页KV

连续KV地址简单、访问连续，但动态扩缩和碎片管理困难。Paged KV用逻辑block表映射到物理block，便于动态容量管理，却增加地址查表和物理不连续。Decode Kernel必须按目标layout做benchmark，不能用连续KV结论替代分页场景。

### 28.5 必做手算

给定模型参数，计算KV容量、每个Decode token读取的最低KV bytes、MHA/GQA/MQA差异，并判断为什么减少Nkv会降低容量和带宽压力。

---

## 29. 扩展教材四：量化与低精度路径

### 29.1 基本公式

对称量化：

```text
q = clamp(round(x/scale), qmin, qmax)
x_hat = q × scale
```

非对称量化再加入`zero_point`。Scale粒度：

- Per-tensor：一个scale，开销小，难适应不同通道范围。
- Per-channel：每输出/输入通道一个scale，精度好，读取/广播增加。
- Per-token：适应激活动态范围，需运行时求scale。
- Per-group：精度与元数据/计算成本折中，常见于低bit weight。

### 29.2 Weight-only与W8A8

Weight-only减少权重读取，activation仍为浮点；W8A8同时量化激活和权重，可使用整数矩阵路径，但需要activation scale、整数累加和反量化/epilogue。

### 29.3 公平实验

必须报告：原始模型dtype、scale粒度、accumulator、输出dtype、是否包含量化/反量化、误差/任务质量和硬件指令路径。量化变快可能来自bytes减少、矩阵吞吐增加或二者共同作用。

---

## 30. 扩展教材五：Dispatch、Autotune与性能回归

### 30.1 为什么一个Kernel不够

大方阵偏好大tile和深pipeline；Decode小M需要更多grid并行；未对齐或尾块需要通用路径。典型调度：

```python
if dtype == fp16 and aligned and M >= 128 and N % 16 == 0 and K % 16 == 0:
    return tensor_core_large_tile(a, b)
elif dtype == fp16 and M <= 16:
    return small_m_kernel(a, b)
else:
    return safe_fallback(a, b)
```

每个条件都必须有正确性测试、代表shape和fallback。

### 30.2 Autotune

搜索空间只放有物理依据的config：tile、num_warps、num_stages等。每个config先过正确性；分离首次JIT；结果绑定GPU架构、软件版本、dtype和shape bucket。Autotune缓存本身有冷启动和维护成本。

### 30.3 回归阈值

先测历史噪声。例如某shape正常P95波动3%，统一1%阈值会制造误报。回归记录环境、代码版本、contract、median/P95、正确性摘要和预期变更。

---

## 31. 50道正常掌握题答案索引

原核心题1～31由第3～10、14～23章的公式、代码、实验和答案覆盖。下面给出扩展题的标准答案；作答仍必须附自己的数据。

32. **Online Softmax：** `m=max(ma,mb)`，`l=la·exp(ma-m)+lb·exp(mb-m)`，所有指数自变量≤0。
33. **Fused/Online：** Fused缓存整行，理想读x一次但占`cols×4B`片上状态；Online只保留统计状态，通常读x两次。
34. **Shape边界：** 少rows导致grid不足；超长cols导致单行片上容量、归约和多block合并问题。
35. **Norm差异：** LayerNorm有mean/variance和gamma/beta；RMSNorm只用平方均值缩放，通常无减均值。
36. **取消误差：** `E[x²]`和`E[x]²`很大且接近时相减损失有效位，Welford避免直接相减。
37. **融合读写：** 画未融合中间Tensor，逐个统计读/写；融合通常消除中间写回和再次读取，实际bytes由Profiler验证。
38. **RoPE索引：** position选择角度，head/token定位向量，pair/dimension选择旋转分量；布局决定配对公式。
39. **Gather/Scatter：** 地址由index决定，不保证warp连续；scatter还存在写冲突和atomic，数据分布决定性能。
40. **Attention阶段：** Prefill矩阵大且可能计算受限；Decode query小、读取长KV，常带宽/并行度受限。
41. **FlashAttention：** 收益核心是避免完整scores/probability的HBM往返，用分块和Online状态换取更少IO。
42. **Fast path：** 条件包含dtype、shape、layout、alignment、架构；分别测试满足、刚好不满足和fallback。
43. **Fallback：** 它可能执行不同Kernel、转换或库调用，时间不代表纯快路径，必须单独报告覆盖率和成本。
44. **Dispatch依据：** dtype、M/N/K或rows/cols、alignment、整齐度、workload大小和GPU架构。
45. **Autotune绑定：** 编译器、GPU资源、指令、cache和shape改变都会改变最优config。
46. **回归阈值：** 阈值需高于该shape历史噪声，并按环境和shape设定；1%可能低于测量波动。
47. **可维护性：** 统一API、测试、benchmark、Profiler入口、dispatch/fallback、环境快照、文档和历史回归。
48. **Epilogue：** 消除bias/activation/residual中间读写和launch；代价是register、分支、layout、精度和config组合。
49. **Scale粒度：** 越细通常精度越好，但scale元数据、读取、求取和广播成本越高。
50. **KV布局：** 连续布局地址简单；分页布局需逻辑→物理查表、访问可能不连续，但动态容量管理更好。

---

## 32. 从不会到会的执行判据

每个模块做四次：

1. **跟随例题：** 能解释每个公式、地址和代码行。
2. **完成原实验：** 正确性和性能数据与预测对照。
3. **完成变式：** 换shape/dtype/tile或融合项，先预测再验证。
4. **延迟闭卷：** 一周后重写核心实现，回答问题并指向自己的Profiler证据。

如果只能看懂答案，记为L1；能运行原例记为L2；能独立完成变式和边界记为L3；能解释反直觉结果、维护dispatch和回归才记为L4。

---

## 33. 阶段2正式口试25题详细答案

本章严格对应第22章25道正式口试。标准不是背术语，而是能把公式、代码版本、shape、Profiler指标和Duration连接成因果链。

### 33.1 M/N/K怎样进入地址和FLOPs

Row-major GEMM：

```text
C[m,n] = Σ(k=0..K-1) A[m,k] × B[k,n]
A地址 = baseA + (m×K+k)×sizeof(dtype)
B地址 = baseB + (k×N+n)×sizeof(dtype)
C地址 = baseC + (m×N+n)×sizeof(output_dtype)
有效FLOPs = 2MNK
```

M决定A/C行数和输出行并行，N决定B/C列数和输出列并行，K是每个输出点积长度。小M可能grid不足；大K增加每输出计算和数据复用机会；非整齐M/N/K分别产生输出尾块和K尾块。

**证据：** 对一个真实shape画出CTA到`m/n`的映射，手算`2MNK`，与benchmark的有效TFLOP/s计算一致。

### 33.2 Naive每输出读取多少，实际DRAM为何更少

一线程一输出的Naive代码通常每个`C[m,n]`循环K次，每次读取一个A和一个B，共`2K`元素load、K次FMA和一次C store。FP32代码级字节约为`8K+4`。

全矩阵相邻线程会重复读取A行或B列，L1/L2 cache和硬件合并可能复用这些请求，因此实际DRAM bytes常小于“每线程load次数×字节”。代码级访问揭示复用机会，Profiler的L1/L2/DRAM绝对bytes显示硬件层实际流量。

**误区：** 用代码级8K字节直接宣称DRAM一定搬了这么多；或反过来因为cache有效就认为Shared tiling没有价值。

### 33.3 TILE16为什么理论减少16倍Global读取

对一个`16×16`输出tile和一个K方向16元素分块：Naive的256个输出各读16个A和16个B，共`256×32=8192`元素load。Tiled版本合作加载一个`16×16` A tile和一个`16×16` B tile，共512元素，然后复用计算256个输出。

```text
理论代码级减少 = 8192/512 = 16倍
```

真实DRAM减少未必16倍，因为Naive可受cache帮助；Tiled还增加shared读写和barrier。最终比较global/L2/DRAM bytes、shared流量和Duration。

### 33.4 Tiled Kernel两个Barrier保护什么

第一个barrier位于合作加载之后，保证block所有线程已把本轮A/B tile写入shared，任何线程才开始读取计算。第二个barrier位于本轮计算之后，保证所有线程已读完当前tile，才允许下一轮覆盖同一shared缓冲。

少第一个会读未完成数据；少第二个会发生“下一轮写覆盖上一轮仍在读”的竞态。尾部线程不能在barrier前不一致return，应写identity/0并参加同步。

### 33.5 当前Tiled GEMM的Padding为什么可能无效甚至变差

Padding解决的不是“Shared Memory天然很慢”，而是同一个warp在同一条Shared指令中访问多个不同地址、这些地址却落到同一bank的问题。必须先根据当前Kernel的lane访问地址判断是否存在冲突，不能因为经典矩阵转置使用`[TILE][TILE+1]`，就直接给GEMM也加一列。

当前`gemm_tiled`使用`16×16`线程块。线性thread id以`threadIdx.x`优先递增，因此warp 0由`ty=0, tx=0..15`和`ty=1, tx=0..15`组成。以常见FP32、32个bank、每bank宽4字节的模型计算：

```text
bank(row, col) = (row × leading_dimension + col) % 32
```

- 合作写入`a_tile[ty][tx]`或`b_tile[ty][tx]`时，`[16][16]`布局中warp 0访问bank 0～31，原本没有冲突。
- 计算读取`a_tile[ty][k]`时，每个半warp读取同一个地址，可走broadcast；两行对应两个不同bank。
- 读取`b_tile[k][tx]`时，相同`tx`的两条lane读取同一地址，其余`tx`落在连续bank，也不是经典转置冲突。
- 改为leading dimension 17以后，合作写入第二行映射到bank 17～31和0；其中第二行最后一个元素与第一行第一个元素可能形成同bank不同地址访问，反而可能产生2-way conflict。

所以本项目里的`[16][17]`不能仅凭经验判定为优化。合格证据链是：

1. 对目标Shared指令列出一个warp的lane、地址和bank编号。
2. 区分同地址broadcast与同bank不同地址冲突。
3. 用NCU比较padding前后的Shared Bank Conflicts、Wavefronts、Shared吞吐和绝对事务量。
4. 最后比较相同shape和相同正确性条件下的Duration；冲突减少但Duration不降，也不能宣称优化成功。

### 33.6 2×2 Thread Tile怎样减少Shared读取，代价怎样验证

每线程累加2×2输出。每个K步只需读取2个A值和2个B值，就产生4次乘加；一线程一输出则每个输出分别读取对应A/B。A值在两个n输出间复用，B值在两个m输出间复用，单位输出的shared load下降。

代价是4个accumulator和更多地址/临时变量增加register；CTA输出tile变大、grid block数减少；尾块浪费可能增大。验证时保持M/N/K、dtype、正确性和计时方法不变，对比tiled与2×2版本的Shared load指令/事务、Registers/Thread、spill、Blocks/SM、Achieved Occupancy、Eligible Warps和Duration。只有Shared读取下降后最终Duration也下降，才能证明收益大于资源代价。

### 33.7 4×4慢于2×2有哪些竞争假设

至少考虑：

1. 16个accumulator增加register，降低blocks/SM或产生spill。
2. CTA输出tile更大，grid变小，小M/N时并行度不足。
3. 不规则shape的无效输出和K尾块浪费增加。
4. 更长依赖链或指令数限制发射。
5. Shared复用收益已不再是主瓶颈。

最小实验分别固定shape、检查编译register/spill；选择整齐大shape排除尾块；比较grid/wave；看eligible warp和stall。不能只凭occupancy下降选中假设1。

### 33.8 Occupancy下降为何不自动等于性能下降

Occupancy是活跃warp容量指标，不是有效工作速度。Register tiling可能降低occupancy，却减少shared/global访问、提高数据复用和ILP，使每个warp工作更高效。只要仍有足够eligible warps隐藏延迟，Duration可以下降。

相反，提高occupancy若导致spill、重复load或更小低效tile，可能变慢。必须比较register、spill、active/eligible warps、issue、bytes和Duration。

### 33.9 Float4到底改变什么

`float4`通常把4条标量load/store表达为一条更宽的向量指令，可能减少指令数和改变request组织。它不改变数学上必须读取/写入的元素数，所以最低bytes不减少4倍；若标量访问本来已合并，sector/DRAM bytes也可能基本不变。

验证：检查基址、leading dimension和尾部满足16-byte alignment；用PTX/SASS确认宽指令；比较指令数、requests、sectors、DRAM bytes和Duration；对未对齐输入验证fallback。

### 33.10 M/N/K非整齐分别怎样处理

- M尾部：输出行越界线程不写C，加载A时用mask/填0。
- N尾部：输出列越界线程不写C，加载B时用mask/填0。
- K尾部：最后K tile只加载有效元素，其余shared/寄存器位置填乘法identity 0。

策略可为Kernel内mask、主体+tail Kernel或多实现dispatch。Tensor Core路径若要求特定倍数，可padding或fallback。任何方案都不能越界或静默丢数据；性能报告要包含尾块利用率。

### 33.11 四层Tile是什么关系

CTA tile是一个block负责的输出区域；其中划分多个warp tile；SIMT路径下warp tile再分为thread tile；Tensor Core路径下warp协作执行一个或多个MMA tile，fragment分布在warp lanes/register中。

```text
输出矩阵 → CTA tile → Warp tile → Thread tile或MMA tile
```

K tile描述每轮加载的归约维分块。合格回答要画自己的版本，标出block维度、warp数、M/N/K方向、A/B加载者和C持有位置，而不是只给定义。

以本文`m16n16k16` WMMA教学版本为例：一个warp协作完成一个`16×16`输出warp tile；一个block包含4个warp，并让4个warp沿M方向排列，因此CTA tile覆盖`64×16`输出；K方向每次消费一个`K=16`的MMA tile，若`K=128`就循环8次。这里不能再说“每个thread固定计算某几个C元素”，因为WMMA fragment怎样分布到32条lane由实现决定。

SIMT Register Tiling则不同。例如一个`16×16`线程块、每线程持有`2×2`输出时，thread tile是`2×2`，CTA输出tile是`(16×2)×(16×2)=32×32`；warp tile由warp中32个thread tile共同组成。这个例子说明四层tile是嵌套的工作划分，不是四个可以独立选择的数字。

### 33.12 FP16、BF16、TF32的差异

必须从“数据怎样存储”和“硬件怎样计算”两个层面回答：

- **FP16（IEEE binary16）：** 1位符号、5位指数、10位显式小数；占2字节。最大有限值为65504，最小正规格化正数为`2^-14`，1附近的间隔约为`2^-10`。它的有效精度高于BF16，但指数范围明显小于BF16/FP32，更容易上溢或下溢。
- **BF16：** 1位符号、8位指数、7位显式小数；占2字节。指数范围接近FP32，但1附近的间隔约为`2^-7`，输入量化误差通常大于FP16。它用较少尾数精度换取了更大的动态范围。
- **TF32：** 通常仍由FP32张量和FP32接口提供数据，内存中仍占4字节；Tensor Core乘法路径保留8位指数，并使用约10位显式小数的乘法精度，随后通常累加到FP32 accumulator。它不是一种可声明为2字节张量的独立存储dtype。

“通常FP32累加”是所选MMA/WMMA/库路径的属性，不是仅由输入dtype自动保证。完整精度契约必须写成：

```text
输入/存储dtype → 乘法实际精度 → accumulator dtype → 输出dtype
```

例如本文WMMA路径是`FP16输入 × FP16输入 → FP32累加 → FP32输出`。即使使用FP32累加，也无法恢复FP16输入在进入Kernel前已经丢失的尾数。验证时应以FP32 reference报告max/mean/P99误差、NaN/Inf数量，并观察误差如何随K和输入范围变化。

### 33.13 怎样证明用了Tensor Core

源码调用WMMA、`tl.dot`或库API只证明“程序员希望使用Tensor Core”，不能证明最终生成的机器代码确实走了Tensor Core。证据强度从弱到强分为四层：

1. **源码意图：** 存在`wmma::mma_sync`、`tl.dot`或相应库调用，只能作为定位入口。
2. **编译条件：** 记录GPU、CUDA版本和目标架构；RTX 4090应确认生成了Ada对应的`sm_89`代码，而不是只保留不匹配的架构产物。
3. **指令证据：** 在NCU Source/Disassembly或反汇编中找到该架构对应的MMA/HMMA矩阵乘加指令族。这是“执行路径确实存在”的直接证据，具体助记符会随GPU代际变化。
4. **运行时证据：** NCU中的Tensor/MMA pipe指令或利用率指标必须出现非零工作，同时Profiler捕获的Kernel名称必须是WMMA快路径，而不是fallback Kernel。

还要做一个SIMT对照，并记录相同shape下的Duration、有效TFLOP/s和数值误差。性能更快只能作为辅助证据：一个Kernel即使变快也不自动证明用了Tensor Core；反过来，看到MMA指令但因shape太小、转换或资源开销而变慢，也不能否认Tensor Core路径存在。

### 33.14 当前WMMA实现为何选择Fallback

WMMA fragment和`load_matrix_sync`按固定M/N/K tile协作执行，所有参与warp的lane还必须以一致控制流进入WMMA操作。本文教学版本使用`m16n16k16`，没有为M/N输出尾块和K尾块准备安全的padding或tail路径；若直接让部分lane跳过或直接加载不完整fragment，可能越界，也可能把无效值参与计算。因此，**当前实现**只在M/N/K满足16倍数及其布局、leading dimension、地址约束时进入WMMA，其他shape路由到正确的通用Kernel。

这不是“数学上的GEMM遇到不规则shape必须fallback”。至少还有三种正确方案：

1. 把A/B/C临时padding到合法tile，WMMA计算后裁剪有效输出；代价是分配、填充、拷贝和无效FLOPs。
2. 用WMMA处理完整的主体区域，再由SIMT tail Kernel处理M/N/K尾部；代价是多Kernel launch和边界组合复杂度。
3. 建立多个特化Kernel或交给cuBLAS/CUTLASS等实现dispatch；代价是实现、编译和调度复杂度。

报告必须把四个时间分开：纯WMMA快路径Kernel时间、Host dispatch开销、输入转换/padding等准备时间、fallback Kernel时间；同时记录真实shape中快路径命中率。不能把fallback的库调用时间标成“手写WMMA性能”，也不能只报告整齐shape而隐藏不规则shape的端到端成本。

### 33.15 Triton配置各控制什么

- `BLOCK_M/N`：一个program计算的输出tile，影响复用、无效尾块和grid数量。
- `BLOCK_K`：每轮归约分块，影响加载粒度、循环次数和片上资源。
- `GROUP_M`：调整program执行顺序，使相邻program更可能复用A/B数据于L2。
- `num_warps`：一个program使用的warp数，影响并行执行和资源。

还应理解`num_stages`控制软件流水深度。配置必须在多shape上实验，不可把单一大方阵最优值推广到小M。

### 33.16 Triton首次调用为何必须分开

Triton首次遇到某组代码/constexpr/config/shape时可能JIT编译、加载Kernel并初始化缓存。这是冷启动成本，不是稳态Kernel执行时间。

性能报告分别记录首次端到端时间和warmup后的CUDA Event统计。若业务关心短生命周期进程，冷启动仍是重要指标，但不能与稳态吞吐混成一个数字。

### 33.17 NSYS和NCU分别回答什么

NSYS看时间线：CPU gap、launch、同步、拷贝、多个Kernel顺序、stream重叠和端到端占比。它先判断问题在Host、系统调度还是某个Kernel。

NCU深入单Kernel：launch配置、寄存器/shared、访存层级、scheduler、stall、源码/指令和Roofline。正确顺序通常是先NSYS定位值得分析的Kernel，再用NCU验证内部假设。

### 33.18 NCU八步问题树

八步不是八个彼此独立的指标，而是一条从结果到原因、再回到结果的排查顺序：

1. **Duration与工作量：** 先确认shape、dtype、数学工作量、正确性、warmup和采样方法一致。否则后面的百分比不可比。
2. **Launch Stats：** 看grid、block、wave数量和每个Kernel实际启动配置，先排除小grid、错误block或比较了fallback的问题。
3. **SM吞吐与Memory吞吐：** 判断计算侧、访存侧或两侧都没有被充分利用，只用于确定下一步调查方向，不直接下“compute-bound/memory-bound”结论。
4. **Memory层级：** 比较DRAM/L2/L1/Shared的绝对bytes、requests、sectors、命中和bank conflict，区分代码load、cache流量和真实DRAM流量。
5. **片上资源与并发：** 查看Registers/Thread、Shared/Block、spill、Blocks/SM和Achieved Occupancy，判断资源是否限制驻留；不能把低Occupancy直接等同于慢。
6. **Scheduler与Stall：** 先看Eligible Warps和Issue Slot是否不足，再分析Top Stall Reason，并把stall映射回依赖、访存、barrier或源码位置。
7. **源码与指令：** 用Source/PTX/SASS确认向量load、MMA、spill、重复转换或额外控制指令是否真的生成和执行。
8. **Roofline宏观边界：** 明确FLOPs和bytes口径，用Arithmetic Intensity、带宽roof和计算roof判断理论上限方向，并检查前面形成的因果解释是否合理。

完成第8步后必须回到第1步的Duration，用最小代码改动复测。这个“回到Duration”是验证闭环，不作为第9步。顺序的意义是避免一开始挑一个高百分比讲故事；Roofline也是条件性宏观检查，不是自动最终判决。

### 33.19 DRAM Throughput上升但为何变慢

Throughput百分比是“相对峰值使用程度”，不表示完成了更多有效工作。新版本可能搬运了更多实际bytes，因而显存更忙、时间却更长；也可能计算/同步让总周期增长，DRAM活动占比随之变化。

必须同时比较有效工作量、绝对DRAM bytes、requests/sectors、Duration和其他资源。若有效FLOPs相同、bytes增加且Duration增加，通常不是更好的访存优化。

### 33.20 Stall为何先看Eligible Warp和Issue Slot

Stall表示某warp当前不能发射的原因分布，但只有scheduler缺少eligible warp、issue slot实际空闲时，stall才可能限制吞吐。某类stall百分比高也可能发生在本来就有其他warp可发射的场景。

先看active/eligible warps和issue效率，确认发射不足；再把主要stall联系到register依赖、访存、barrier、grid和源码；通过最小实验改变该原因并观察Duration。

### 33.21 Roofline三要素

横轴是Arithmetic Intensity（FLOPs/byte），纵轴是性能（FLOP/s）；斜线带宽roof为`bandwidth×AI`，水平线是计算峰值。交点对应：

```text
ridge_point = peak_compute / peak_bandwidth
```

AI低于交点倾向带宽上限，高于交点倾向计算上限。必须写明FLOPs定义、dtype和bytes层级。Roofline不包含小grid、launch、依赖和所有指令混合，因此只给上限方向。

### 33.22 有效TFLOP/s与实际执行指令

有效TFLOP/s按数学工作量`2MNK/time`计算，便于比较完成同一GEMM的速度。实际执行可能包含尾块padding、无效FMA、地址/控制指令、转换或不同Tensor Core指令，一条MMA也代表很多数学FLOPs。

所以有效TFLOP/s不等于“指令计数×某固定数”，也不表示无效工作少。需要结合执行指令、tile利用率和有效shape解释。

### 33.23 Online Softmax怎样合并且稳定

两段状态分别为最大值和相对归一化和`(m1,l1)`、`(m2,l2)`：

```text
m = max(m1,m2)
l = l1×exp(m1-m) + l2×exp(m2-m)
```

因为`m≥m1,m2`，对有限输入有`m1-m≤0`、`m2-m≤0`，从而避免对大正数直接`exp`。最终输出为`exp(x-m)/l`。逐元素更新其实是同一合并公式：把旧状态`(m_old,l_old)`与单元素状态`(x,1)`合并即可；分块、warp和block归约也都使用相同的结合规则。

空分段可把`(-inf,0)`作为逻辑identity，但实现不能无条件计算两个空identity的`exp(-inf-(-inf))`，因为该表达式会产生NaN。安全做法是携带valid标志，或在`l==0`时直接返回另一侧状态；两侧都空则直接返回identity。

特殊值必须先定义契约：

- 有限输入应得到有限、非负输出，且每行和接近1。
- 输入含NaN时，通常按reference传播NaN，不能把NaN静默变成普通概率。
- 一行全为`-inf`时，数学上分母为0，常见reference会产生NaN；测试应明确跟随目标reference，而不是误判为数值不稳定Bug。
- 输入含`+inf`时，`inf-inf`也需要明确语义；若要实现“多个`+inf`均分概率”，必须写成额外特化，不能声称来自普通stable-softmax公式。

**证据：** 先用两个非空有限分段手算，与一次性stable softmax比较；再覆盖空分段、FP32/FP16/BF16、整体平移、极值、NaN/Inf和非2次幂列数，并记录max abs/rel error、行和误差和首个特殊值位置。

### 33.24 本文Fused Softmax为何可能慢于Online

这里的两个名称只指本文的两个具体实现，并不是说“fused”和“online”在一般概念上互斥：

- `softmax_fused`把一整行输入转换成FP32后缓存到动态Shared Memory，随后从Shared完成max、exp-sum和输出。它通常只从Global读取输入一次，但Shared需求约随`cols×sizeof(float)`增长。
- `softmax_online`不缓存整行，只为每个thread/warp保留`(m,l)`统计状态；第一次遍历输入得到全行状态，第二次再从Global读取输入并写出归一化结果。

因此二者交换的是“少一次Global读取”和“更少片上存储”：

1. 行较长时，`softmax_fused`的动态Shared可能降低Blocks/SM和Achieved Occupancy，甚至超过容量限制。
2. Shared版本还会增加Shared读写、同步和bank访问；少一次Global读取并不等于总指令或总周期一定更少。
3. 行较短但rows很少时，两者都可能受launch和grid不足主导，内存遍历次数不是主因。
4. exp吞吐、reduction依赖链、Registers/Thread和Eligible Warps也可能成为真正限制。

实验必须在相同shape、dtype和正确性条件下比较Global/DRAM/L2绝对bytes、Shared bytes/事务、Shared/Block、Blocks/SM、Registers/Thread、Eligible Warps、Top Stall Reason和Duration。不预设Online一定更快，也不能只凭“Global读取一遍或两遍”下结论。

### 33.25 最佳GEMM在哪些Shape不适用

这题必须引用自己的最佳版本。常见限制包括：小M导致大CTA tile的grid不足；M/N/K不规则导致mask和无效计算；N/K不满足向量或WMMA倍数进入fallback；小矩阵受launch主导；极长条shape的数据复用方向不同；非对齐地址关闭float4；目标GPU代际改变资源和指令路径。

下面是一份作答模板。尖括号内容必须替换成自己的实测数据，不能把示例预测当成项目结论：

```text
我的最佳版本：<版本名>
它在 <基准shape/dtype> 上的Duration为 <实测值>，
相对 <对照版本> 的加速为 <实测值>。

失败shape 1：M=16, N=4096, K=4096
预测：若该版本使用较大的CTA输出tile，grid可能不足以填满RTX 4090的SM。
证据：填写grid/block、waves、SM吞吐、Achieved Occupancy和Duration。
处理：为小M dispatch到较小tile或库实现。

失败shape 2：M=65, N=65, K=17
预测：M/N输出尾块和K尾块使大量thread/FMA无效，有效工作利用率下降。
证据：填写有效FLOPs、实际tile覆盖、Duration、分支/执行线程和load bytes。
处理：选择较小tile、主体+tail Kernel或通用masked实现。

失败shape 3：M=1024, N=1024, K=1025，或构造非16-byte对齐输入
预测：WMMA/float4约束不满足，进入fallback或标量路径。
证据：填写NSYS中的实际Kernel名称、向量/MMA指令是否出现、fallback时间和端到端Duration。
处理：显式dispatch；若padding，则单独报告转换、padding和裁剪成本。
```

合格答案至少列出三个失败shape，并对每个shape完成“运行前预测→Benchmark结果→NCU/NSYS证据→根因→dispatch/fallback”。不能只说“某些小shape不适用”，也不能只列Profiler百分比而没有最终Duration。

### 33.26 复测标准

把25题分成五组：GEMM映射1～6、资源与退化7～10、低精度与技术栈11～16、Profiler17～22、Softmax与边界23～25。一周后每组随机抽两题；至少9/10能闭卷回答，且每组至少一题能给出自己项目的具体shape、数值和报告路径。只会复述标准答案、没有项目数据时，最高记为L2，不能作为阶段2完成证据。
