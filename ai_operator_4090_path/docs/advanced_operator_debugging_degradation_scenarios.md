# GEMM、Softmax、LayerNorm、RMSNorm 与 Attention 高阶调试技巧及恶化场景

> 适用项目：`ai_operator_4090_path`
>
> 目标设备：单张 NVIDIA RTX 4090（Ada，SM 8.9）
>
> 文档定位：在基础正确性测试通过后，用可证伪假设、单变量恶化实验和 profiler 证据定位深层正确性、性能及工程问题。

## 1. “恶化场景”与调试目标

本文中的“恶化”包括三类：

1. **正确性恶化**：输出出现 NaN、Inf、越界、非确定性，或误差随 shape、数值范围、序列长度显著放大。
2. **性能恶化**：同一输入、dtype、实现和环境下，kernel `Duration` 回退；或者 kernel 本身未退化，但端到端延迟因拷贝、转换、同步、launch gap 增加。
3. **路径恶化**：优化路径因尾部、对齐、dtype 或 shape 条件退回 fallback；接口名没有变化，但真正执行的 kernel 或前后处理已经变化。

调试不是“找到一个看起来异常的指标”，而是建立以下闭环：

```text
稳定复现 → 确定异常层 → 提出至少两个候选假设 → 单变量实验
        → 工具或源码证据证伪 → 修复 → 最小回归 → 全量回归
```

本文覆盖当前项目导出的 7 个 GEMM、4 个 Softmax、4 个 LayerNorm、5 个 RMSNorm 和 4 个 Attention/KV-cache 接口。别名或转发关系不算独立优化，例如 `attention_causal_naive` 仍复用 naive causal 路径。

## 2. 调试前必须固定的变量

任何性能结论至少固定并记录：

- Git commit、编译参数、CUDA/PyTorch/驱动版本、GPU 名称与 capability。
- 算子接口和实际 kernel 名；不能只写“GEMM”或“Attention”。
- 完整 shape、dtype、layout、stride、对齐、causal、`eps`、`kv_len`。
- 随机 seed、输入分布及 reference；极值输入与普通随机输入必须分开记录。
- warmup 次数、测量次数、CUDA stream、是否同步、是否包含输入转换和内存分配。
- GPU 时钟、温度、功耗及是否有其他进程争用；怀疑环境抖动时先重复基线。

仅当 workload 完全相同，才可比较两个版本的 `Duration`。Attention Prefill 和 Decode、不同 `kv_len`、不同 hidden size 的数据不能直接排名。

## 3. 先确定异常位于哪一层

| 异常层          | 典型信号                                     | 首选手段                                 |
| --------------- | -------------------------------------------- | ---------------------------------------- |
| 输入/契约层     | dtype、device、shape、stride 错误或静默 copy | 断言、`TORCH_CHECK`、最小输入            |
| 构建/加载层     | 改源码后行为不变或符号缺失                   | 加载路径、mtime、导出符号、`DBG-E01~E04` |
| launch 层       | invalid configuration 或立即报错             | launch check、`DBG-L04`                  |
| 执行/内存层     | OOB、竞态、未初始化读取或异步报错            | Compute Sanitizer、独立进程              |
| kernel 正确性层 | 只在数值极值或边界 shape 出错                | 差分、变形、边界测试和中间量             |
| kernel 性能层   | 固定 workload 的 kernel 时间回退             | CUDA Event、NCU                          |
| 调用链性能层    | kernel 稳定但接口或请求变慢                  | NSYS、NVTX、CPU/CUDA 时间线              |
| 环境层          | 所有版本同时变慢且波动大                     | 基线、时钟、功耗、温度和进程争用         |

不要跨层误用工具：契约错误不先跑 NCU；host copy/sync 不用 kernel 指标解释；OOB 后不复用已损坏的 CUDA context。

### 3.1 工具选择顺序

```text
契约断言与 reference
  └─ 有 CUDA 错误：Compute Sanitizer
  └─ 结果错误：差分/变形/边界/数值测试
  └─ 端到端变慢：NSYS 确认时间归属
       └─ 主要时间在目标 kernel：CUDA Event + NCU
       └─ 主要时间在 copy/cast/sync/launch gap：修调用链
```

进入 NCU 后按固定顺序阅读：

```text
Duration → Compute/Memory → DRAM → L2 → Occupancy
         → Registers/Thread → Top Stall → 回到 Duration 与正确性
```

`Duration` 是固定 workload 的结果，其余指标是解释结果的证据。缺失值必须写 `N/A` 和原因，不能填成测量值 `0`。

## 4. 跨算子的高阶调试技巧

### 4.1 差分测试不只比较一个最大误差

同时记录：

- `max_abs_error`：暴露绝对偏差。
- `max_rel_error`：暴露小量级输出的比例偏差，但分母需加安全下界。
- mismatch 的索引、reference 值、实际值和所在 row/head/token。
- NaN/Inf 数量；NaN 会让普通 `max()` 失去判断意义。
- 误差随归约长度、幅值、dtype、causal 或 `kv_len` 的变化曲线。

若误差只随归约长度单调放大，优先检查累加精度、归约顺序和重标度；若只发生在最后一个 tile，优先检查边界谓词和尾部处理。

### 4.2 变形测试验证“关系”，而不只验证固定答案

| 算子      | 变形关系                                      | 能暴露的问题                   |
| --------- | --------------------------------------------- | ------------------------------ |
| GEMM      | `A @ 0 = 0`、分块结果可拼接、单位阵不改变输入 | 索引、尾部、累加、转置错误     |
| Softmax   | 每行加同一常数后结果不变；行和约为 1          | 漏减最大值、跨行污染、归约错误 |
| LayerNorm | 输入每行加常数时，归一化主体应近似不变        | 均值、方差、广播维度错误       |
| RMSNorm   | 输入乘正数时，在 `eps` 可忽略区间方向基本不变 | 错减均值、平方均值或缩放错误   |
| Attention | 同时置换 K/V token 后非 causal 输出不变       | score 与 V 索引不一致          |

变形关系有适用边界。例如 RMSNorm 在 `eps` 主导的小输入区间并不严格尺度不变；必须在测试报告中写清前提。

### 4.3 用 shape 相图找“断崖”，不要只测一个标准尺寸

至少围绕以下边界扫描 `boundary-1 / boundary / boundary+1`：

- warp：31、32、33。
- 常见 block/tile：127、128、129；255、256、257。
- 向量宽度：不能整除 2 或 4 的维度，以及 8/16 字节地址错位。
- WMMA：16 的倍数及其相邻值。
- Attention：`S=1`、tile 边界、`kv_len=1`、`kv_len=max`。

若时间或误差在边界处跳变，应先确认实际执行路径，再分析该路径的指标。平滑变化更像工作量变化；突然跳变通常意味着 fallback、额外 launch、尾部 kernel 或资源阈值变化。

### 4.4 用 poison、canary 和重复执行暴露隐藏状态

- 输出预填 NaN 或固定 poison 值，检测是否有元素从未被写入。
- 在合法区间前后放 canary，配合 memcheck 检测越界写。
- 同一输入重复执行 20 次并比较 bitwise 或误差分布，暴露竞态和未初始化读取。
- 改变无关内存分配顺序；若结果随 allocator 状态变化，优先怀疑 OOB 或未初始化内存。
- 故意在另一个 stream 上生产输入，用 Event 建立和移除依赖，验证 current-stream 语义。

### 4.5 二分路径而不是一次修改多处

为优化 kernel 保留一个正确、简单的实现作为 oracle。按以下维度逐一切换：

```text
标量 load ↔ 向量 load
共享内存 ↔ 直接全局内存
block reduction ↔ warp reduction
普通 softmax ↔ online softmax
FP32 累加 ↔ 低精度累加
优化路径 ↔ 明确记录的 fallback
```

一次只改变一项。若同时改变 tile、向量宽度、归约算法和 dtype，profiler 只能展示相关性，无法证明根因。

### 4.6 证明真实执行路径

- 用 NSYS 确认接口前后是否出现 cast、contiguous、allocation、memcpy 或同步。
- 用 NCU kernel filter 确认采集的是目标 kernel 和正确 invocation。
- 对 Tensor Core 声明必须检查 NCU Source/SASS 中的 MMA 指令；接口名含 `wmma` 不构成证据。
- 用 aligned、misaligned、tail 三组输入对照，确认向量化路径和 fallback。若 dispatch 到不同 kernel，用 NSYS 核对 kernel 名；若同一 kernel 通过参数选择标量/向量 load，则用源码、Source/SASS 和 load 指标确认内部路径。
- 对重复/转发接口，先查源码调用链；不要把相同 kernel 的两个包装函数当成两种优化。

## 5. GEMM 高阶调试与恶化场景

核心不变量为 `C[M,N] = A[M,K] @ B[K,N]`。先判断问题是否集中在 M、N 或 K 的尾部，再判断是 load、累加还是 store 阶段。

| 恶化场景             | 定位信号                              | 首选证据与动作                                |
| -------------------- | ------------------------------------- | --------------------------------------------- |
| K 尾部遗漏           | K 跨 tile 后，部分列误差放大          | mismatch 热图；越界 load 置零                 |
| M/N 尾部越界         | 最后一行或列错误，偶发 illegal access | memcheck、canary；分别限制 load/store         |
| shared-memory 竞态   | 同一输入重复执行，结果偶发变化        | racecheck/synccheck；补齐 CTA 同步            |
| bank conflict 恶化   | Duration 上升而 DRAM 变化小           | bank conflict、Top Stall；调整 shared layout  |
| register tile 过大   | occupancy 降低或出现 local spill      | Registers、local load/store；缩小 thread tile |
| float4 load 失效     | B 错位或 N 尾部时收益消失             | Source/SASS、load 指标；保留标量路径          |
| WMMA 端到端变慢      | WMMA kernel 快，但接口时间增加        | `DBG-L03`、NSYS；消除或计入 dtype 转换        |
| Tensor Core 未执行   | Tensor Pipe 或 MMA 指令无证据         | Source/SASS、实际 kernel；检查 dispatch       |
| 低精度误差恶化       | 误差随 K 和动态范围持续放大           | FP32/FP64 reference；保持 FP32 accumulator    |
| 隐藏 contiguous copy | kernel 稳定，但接口时间增加           | `DBG-L01`、NSYS；拒绝或显式计入 copy          |
| 小矩阵 launch-bound  | kernel 很短，吞吐仍低                 | NSYS launch gap；采用 batching 或 fusion      |

### 5.1 GEMM 专项技巧

- 制作 mismatch 坐标热图：整块 16/32 周期错误通常指向 tile 索引；仅最后一条带错误通常指向尾部谓词。
- 将 A 设为单位阵或行编码，将 B 设为列编码，可以直接从错误值反推出取错的行/列。
- 为每个 tile 阶段分别检查 A load、B load 和 C store，避免把所有索引压在一个最终输出上猜测。
- 比较 `gemm_tiled`、`gemm_tiled_padding`、`gemm_regtile2x2`、`gemm_regtile4x4` 时只用同一 M/N/K；先看 Duration，再解释资源交换。
- WMMA 对非 16 倍数 shape 的 fallback 必须同时验证正确性和路径；不能把 fallback 的结果记成 Tensor Core 性能。

推荐现有实验：`GEMM-C01~C03`、`GEMM-D01`、`GEMM-P01~P04`、`DBG-L01`、`DBG-L03`、`DBG-L04`。

## 6. Softmax 高阶调试与恶化场景

稳定 Softmax 每行满足：

```text
m = max(x)
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

除 reference 误差外，还应检查每行输出非负、有限，且行和约为 1。

| 恶化场景              | 定位信号                           | 首选证据与动作                             |
| --------------------- | ---------------------------------- | ------------------------------------------ |
| 未减最大值            | 大正数输入出现 Inf/NaN             | 记录 max/sum；使用 max-shift               |
| 最大值 identity 错    | 全负数行输出异常                   | 对照 `-inf` identity                       |
| sum identity/尾部错误 | 31/32/33 列附近行和不为 1          | 行和、lane mask；屏蔽无效 lane             |
| warp mask 错误        | 非完整 warp 才出错                 | active mask；使用真实 ballot mask          |
| 跨行 shared 污染      | 某行结果依赖相邻行                 | 单行/双行对照、racecheck                   |
| online 重标度遗漏     | 后段出现新 max 时结果错误          | running max/sum；重标历史状态              |
| 全 mask 行 NaN        | mask 后整行无有效元素              | 检查 denominator；定义全 mask 语义         |
| NaN 传播不一致        | 自定义实现与 PyTorch 语义不同      | 记录 NaN 位置；明确传播契约                |
| online 性能反退       | 读写减少但 Duration 增大           | Registers、Top Stall、exp；按列宽 dispatch |
| 小 row launch-bound   | 短 kernel 的版本差异被 launch 淹没 | NSYS、batch 扫描；fusion                   |

### 6.1 Softmax 专项技巧

- 对每一行分别记录 `max`、分母与输出和，先定位 max 阶段还是 sum 阶段。
- 用“最后一个元素成为新最大值”的序列专门测试 online softmax 重标度。
- 用统一加常数的变形测试检查稳定实现；若输出明显变化，通常不是普通浮点舍入。
- 将列宽按 warp、block 和尾部边界扫描，建立实现选择相图，而非默认 warp 版本总是更快。
- 当 `exp` 依赖链成为瓶颈时，occupancy 提高不保证 Duration 下降；必须回到同 workload 时间验证。

推荐现有实验：`SM-C01`、`SM-C02`、`SM-D01`、`SM-P01~P03`、`DBG-U01`。

## 7. LayerNorm 高阶调试与恶化场景

对每一行 hidden dimension：

```text
mean = sum(x) / H
var  = sum((x - mean)^2) / H
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

需分别验证统计量、归一化主体和仿射变换，不能只看最终输出。

| 恶化场景            | 定位信号                        | 首选证据与动作                           |
| ------------------- | ------------------------------- | ---------------------------------------- |
| 方差消减误差        | 大均值、小方差时误差或 NaN 激增 | mean/var、FP64 reference；Welford/两遍法 |
| 常量行不稳定        | 常量行输出非有限或偏离 beta     | 检查 `var+eps`；统一 eps 语义            |
| eps 语义错误        | 仅小方差区误差大                | eps sweep；固定 `sqrt(var+eps)`          |
| gamma/beta 广播错   | 输出呈周期性列错误              | 列编码参数、mismatch 列模式              |
| warp 归约丢 lane    | H=31/32/33 附近统计量偏小       | mean/var、lane mask；修正二级归约        |
| shared 归约竞态     | 同一输入重复执行，输出偶发变化  | racecheck；修正 CTA 同步                 |
| float4 路径退回     | 地址错位或尾部 H 出现速度断崖   | NSYS kernel 名；记录 dispatch            |
| 向量化寄存器恶化    | Memory 改善，但 Duration 上升   | Registers、local memory；缩短 live range |
| 极短 H launch-bound | kernel 优化无端到端收益         | NSYS；与 bias/residual 融合              |

### 7.1 LayerNorm 专项技巧

- 将 kernel 暂时改为输出 mean 和 variance，或用调试分支单独写出统计量；最终误差无法区分统计归约与 affine 错误。
- 使用列编码 gamma/beta：`gamma[j]=j`、`beta[j]=-j` 的小尺寸测试能快速定位广播索引。
- 对“大偏置加小扰动”输入比较 naive variance、两遍算法和 Welford，观察误差随偏置幅度变化。
- aligned、misaligned 和 tail 必须确认实际执行路径；本项目 LayerNorm 会从 vectorized kernel 退回 warp kernel，可由 NSYS kernel 名直接识别。路径退回是正确性契约的一部分，也是性能结论的一部分。
- LayerNorm 比 RMSNorm 多 mean reduction 和中心化，二者不能仅因 shape 相同就预期相同 Duration。

推荐现有实验：`LN-C01~C03`、`LN-P01~P03`、`FI-NORM-01~02`。

## 8. RMSNorm 高阶调试与恶化场景

RMSNorm 不减均值：

```text
ms  = sum(x²) / H
y   = x / sqrt(ms + eps) * gamma
```

如果实现输出接近零均值，需要首先排查误用了 LayerNorm 公式。

| 恶化场景             | 定位信号                           | 首选证据与动作                        |
| -------------------- | ---------------------------------- | ------------------------------------- |
| 错误减均值           | 常量正输入输出接近 0               | 手算 RMS reference；删除中心化        |
| 平方溢出             | 大幅值时 ms 为 Inf                 | ms、NaN/Inf 计数；使用 FP32 累加      |
| 小输入由 eps 主导    | 尺度变形测试在近零区失效           | input scale/eps sweep；注明适用区间   |
| eps 位置错误         | 仅小 ms 区域误差大                 | 手算与 eps sweep；使用 `sqrt(ms+eps)` |
| gamma 索引错         | 输出呈周期性列错误                 | 列编码 gamma；检查 vector lane 索引   |
| float2/float4 混淆   | 报告路径与实际 kernel 不同         | kernel 名、地址模 16、Source          |
| float4 fallback 断崖 | H 或地址跨边界后突然变慢           | 对齐/尾部相图；细化 dispatch          |
| 内存流量主导         | occupancy 高，但 Duration 不再下降 | Memory/DRAM、bytes/element；减少往返  |
| 重复别名误判优化     | 两个接口指标几乎完全相同           | 源码调用链；不计作独立优化            |

### 8.1 RMSNorm 专项技巧

- 先用全 1、正负交替、单脉冲三类可手算输入检查是否错误中心化。
- 分别记录 sumsq、mean square、inverse RMS 和 affine 后输出，定位平方归约与缩放阶段。
- float2 与 float4 对照必须同时固定有效字节数、shape 和 kernel 路径；接口名不能替代实际路径证据。
- 使用 `H=2k±1`、`H=4k±1` 和地址错位组合，区分“维度尾部”与“指针对齐”两个条件。
- 若目标是端到端 Transformer 性能，优先评估 residual-add + RMSNorm 融合，而非只追求孤立 kernel occupancy。

推荐现有实验：`RMS-C01~C03`、`RMS-P01~P03`、`FI-NORM-01~03`。

## 9. Attention 高阶调试与恶化场景

基本语义为：

```text
scores = Q @ K^T / sqrt(D)
scores = apply_mask(scores)
P      = softmax(scores)
O      = P @ V
```

Attention 调试应沿 `QKᵀ → scale/mask → row max/sum → probability → PV` 分阶段定位。只看最终 O 会把 GEMM、mask、Softmax 和 V 索引问题混在一起。

| 恶化场景               | 定位信号                            | 首选证据与动作                           |
| ---------------------- | ----------------------------------- | ---------------------------------------- |
| 缩放遗漏或重复         | D 增大时分布过尖或过平              | 分阶段 score；统一 `1/sqrt(D)` 位置      |
| QK 转置/步长错         | token/head 呈规律性错位             | 小矩阵手算、索引热图；明确 layout        |
| causal off-by-one      | 对角 token 被屏蔽或看到未来         | S=1/2/3 概率矩阵；修正 `<`/`<=`          |
| tile 尾部 mask 错      | 只在 S 非 tile 倍数时错误           | 边界扫描、memcheck；无效 score 设 `-inf` |
| online 重标度错        | 后续 tile 出现新 max 后输出偏差     | tile max/sum/O；重标历史状态             |
| probability 对但 O 错  | P 对齐 reference，最终 O 错位       | 保存 P、检查 P@V 与 V 索引               |
| `kv_len` off-by-one    | 漏掉最新 token 或读取未写槽位       | kv_len 扫描、cache canary                |
| KV stale/uninitialized | 输出随分配状态变化或偶发 NaN        | initcheck、Event、poison；限制有效范围   |
| Prefill 内存爆炸       | 时间和显存随 S² 激增                | NSYS、峰值显存；避免物化 S² 中间量       |
| Decode 低利用率        | 小 kernel 主导，吞吐和 occupancy 低 | NSYS、batch/kv_len 曲线；batching/fusion |
| causal 特化未生效      | causal 与通用路径同速或更慢         | kernel 名、branch、Source；检查 dispatch |
| Prefill/Decode 误比较  | 用不同 workload 得出速度排名        | 分开固定 shape 建立两类曲线              |

### 9.1 Attention 专项技巧

- 小矩阵先手算：`B=H=1`、`S=2/3`、`D=2/4`，令 V 每个 token 各不相同，可直接判断概率选中了哪个 token。
- 使用 one-hot Q/K 让 score 结构可预测；使用 token 编码 V 让最终输出可反推概率索引。
- 对 online attention，专门构造“每个后续 tile 都产生更大 max”的输入，覆盖反复重标度。
- 对 causal attention 检查未来 token 概率严格为 0，同时检查对角线未被误屏蔽。
- Decode 必须扫描 `kv_len=1、tile-1、tile、tile+1、最大容量`，并验证最新写入 token 确实可见。
- 先由 NSYS 判断是否存在 score materialization、内部 contiguous/cast、多个小 kernel 或同步，再决定是否进入单 kernel NCU。

推荐现有实验：`AT-C01~C04`、`AT-P01~P04`、`FI-AT-01~03`、`DBG-S02~S03`。

## 10. 可执行的单变量恶化实验矩阵

以下矩阵用于训练定位能力。标记“现有入口”的场景可直接运行；“临时分支”应在独立临时分支或 `debug_labs` 中注入，不能污染正式导出实现。

| 编号        | 单变量恶化                    | 预期证据                  | 状态               |
| ----------- | ----------------------------- | ------------------------- | ------------------ |
| ADV-GEMM-01 | B 改为 16 字节错位            | 标量 load、Source/SASS    | 现有 profile 场景  |
| ADV-GEMM-02 | N 从 2048 改为 2047           | 尾部 load、正确性、NCU    | 现有 profile 场景  |
| ADV-GEMM-03 | WMMA 输入 FP16 改为 FP32      | cast、临时分配、`DBG-L03` | 现有入口           |
| ADV-GEMM-04 | 删除 tile barrier             | 非确定性、racecheck       | 临时分支           |
| ADV-SM-01   | 删除 max-shift                | Inf/NaN、极值差分         | `DBG-U01` 同类入口 |
| ADV-SM-02   | 列数改为 33                   | 行和、active mask         | 现有测试覆盖       |
| ADV-SM-03   | 新 max 时不重标旧 sum         | running max/sum 错误      | 临时分支           |
| ADV-LN-01   | 改为大偏置小扰动输入          | mean/var、FP64 reference  | 现有实验覆盖       |
| ADV-LN-02   | aligned H 改为尾部 H          | vectorized 退回 warp      | 现有 profile 场景  |
| ADV-LN-03   | gamma/beta 改为列编码         | mismatch 列模式           | 临时输入即可       |
| ADV-RMS-01  | 错误加入减均值                | 常量行错误归零            | 临时分支           |
| ADV-RMS-02  | 16 字节对齐改为 8 字节        | float4 退到 float2        | 现有 profile 场景  |
| ADV-RMS-03  | 输入改为极大幅值              | sumsq 溢出                | 临时输入即可       |
| ADV-AT-01   | `key<=query` 改为 `key<query` | causal 对角线消失         | 临时分支           |
| ADV-AT-02   | 后一 tile 设置更大 score      | online 重标度错误         | 临时输入即可       |
| ADV-AT-03   | `kv_len` 加 1                 | initcheck、poison         | 临时分支           |
| ADV-AT-04   | 移除 cache 写后读 Event       | `DBG-S02`/`DBG-S03`       | 现有入口           |
| ADV-ENG-01  | 增加 hidden contiguous        | copy、`DBG-L01`           | 现有入口           |
| ADV-ENG-02  | 每次调用增加 synchronize      | CPU 阻塞、`DBG-L02`       | 现有入口           |
| ADV-ENG-03  | 索引改为 `index==count`       | invalid write、`DBG-T01`  | 现有入口           |

故障 kernel 必须在独立 Python 进程运行。OOB 或 illegal address 后不要继续复用当前 CUDA context。

## 11. 项目中的执行入口

在 RTX 4090 服务器进入项目根目录后，显式选择能导入 CUDA PyTorch 的解释器：

```bash
cd /path/to/ai_operator_4090_path
export PYTHON_BIN=/path/to/cuda-python

"$PYTHON_BIN" - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

按成本从低到高执行：

```bash
# Gate 0/1：环境、构建、正确性
bash scripts/15_verify_4090.sh smoke

# 选择一个算子专项实验，例如
bash scripts/run_operator_experiment.sh GEMM-C02
bash scripts/run_operator_experiment.sh SM-C02
bash scripts/run_operator_experiment.sh LN-C02
bash scripts/run_operator_experiment.sh RMS-C02
bash scripts/run_operator_experiment.sh AT-C03

# 通用调试训练
bash scripts/run_debug_experiment.sh preflight
bash scripts/run_debug_experiment.sh DBG-L01
bash scripts/run_debug_experiment.sh DBG-T01

# smoke 全部通过后才运行完整验收
bash scripts/15_verify_4090.sh full
```

NCU 报告需要保留 workload、kernel 和 invocation 的映射：

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family all \
  --output-dir reports/ncu_summary \
  --strict
```

本机没有 RTX 4090、CUDA、NCU 或 CUDA PyTorch 时，只能执行 Markdown、脚本和代码静态检查；不能把静态通过写成 GPU 验收通过。

## 12. 一次完整根因复盘的记录模板

```markdown
### [算子]-[编号]：一句话现象

- 环境与 commit：
- 接口 / 实际 kernel：
- shape / dtype / layout / alignment：
- seed / 输入分布 / reference：
- 恶化前基线：
- 恶化后现象：
- 是否稳定复现：
- 首个异常层：输入 / 构建 / launch / 执行 / 数值 / kernel性能 / 调用链 / 环境
- 假设 A：
- 假设 B：
- 单变量实验：
- 排除证据：
- 根因证据：
- 修复：
- 最小回归：
- 同族全量回归：
- Sanitizer / NSYS / NCU 产物：
- 仍未覆盖的边界：
```

性能复盘至少同时报告以下两项：

1. **端到端时间**：是否包含输入转换、分配、拷贝和同步。
2. **目标 kernel Duration**：固定 workload 下 kernel 本体是否真的退化。

二者方向不一致时，以 NSYS 时间线解释差额，不能把所有回退归因于 kernel。

## 13. 验收停止线

每个算子族达到以下条件后，本轮高阶调试整理即可停止，不再无限增加案例：

- 至少完成 1 个数值恶化、1 个边界/路径恶化和 1 个性能归层案例。
- 每个案例都有固定 seed、最小 shape、两个候选假设和至少一项证伪证据。
- 正确性问题有 reference、边界回归和同族全量回归。
- 内存/同步问题有对应 Sanitizer 证据，并在独立进程中复现。
- 性能问题同时给出端到端时间、kernel Duration 和实际 kernel 路径。
- NCU 结论遵循固定八指标顺序，缺失数据保留 `N/A` 原因。
- Tensor Core、向量化或 fallback 结论有 Source/SASS、kernel 名或时间线证据，不能仅由接口名称推断。

## 14. 与现有资料的关系

- 全部正式接口、分级闸门、实验编号和结果表：`docs/operator_validation_experiments_4090.md`
- NCU 固定八指标详解：`docs/ncu_gemm_analysis/ncu_eight_key_metrics.md`
- NCU 未知 kernel 分析：`docs/ncu_gemm_analysis/ncu_unknown_kernel_analysis_workflow.md`
- 隔离故障实验：`debug_labs/README.md`
- Python 到 kernel 的 profile 入口：`benchmark/profile_entry.py`

本文负责“如何构造恶化、如何提出和证伪假设”；现有全算子手册负责“如何执行统一验收与回填结果”。两者配合使用，不重复把静态检查当作 RTX 4090 实测。
