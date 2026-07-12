# NCU GEMM 第二组：`gemm_tiled` vs `gemm_tiled_padding`

这份文档用于第二组 GEMM 对照实验。先填写八个关键指标，再使用 shared-memory 指标验证 bank conflict。

本轮只回答三个问题：

1. `gemm_tiled_padding` 的 Duration 是否稳定优于 `gemm_tiled`？
2. shared memory 的 `+1 padding` 是否改变了主要硬件行为？
3. 如果时间发生变化，bank-conflict 数据能否解释这个变化？

八个指标的定义见：[NCU 八个关键指标详解](ncu_eight_key_metrics.md)。

## 代码唯一主要变化

两个 kernel 的 block、grid、计算循环和同步位置相同，主要区别是 shared-memory 数组的第二维：

```cpp
// gemm_tiled
__shared__ float As[16][16];
__shared__ float Bs[16][16];

// gemm_tiled_padding
__shared__ float As[16][17];
__shared__ float Bs[16][17];
```

shared memory 的理论分配量：

```text
gemm_tiled         = 2 * 16 * 16 * 4 B = 2048 B / block
gemm_tiled_padding = 2 * 16 * 17 * 4 B = 2176 B / block
增加量             = 128 B / block
```

padding 改变了 shared memory 每一行的地址步长，因此可能改变 warp 访问 shared-memory bank 的映射。但是 `+1 padding` 不是必然有效，必须通过 NCU 数据验证。

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

## 运行前先写预测

不要先看 `gemm_tiled_padding` 的结果。先在下面保留自己的预测：

```text
预计 Duration：
预计 Registers / Thread：
预计 Achieved Occupancy：
预计 Top Stall Reason：
预计 bank conflict：
预测理由：
```

本轮至少保留两个竞争假设：

```text
H1：原 tiled 存在 bank conflict。
    padding 后 conflict、相关 stall 和 Duration 应下降。

H2：原 tiled 已经基本没有 bank conflict。
    padding 不会改善 Duration，甚至可能改变原本合理的 bank 映射。
```

## 八个关键指标记录

在 NCU 中分别选择第 6 次稳定执行的 `gemm_tiled_kernel` 和 `gemm_tiled_padding_kernel`。先填写界面中的原始名称和数值，不急着写结论。

| 指标                    | NCU `Details` 中的位置                     | `gemm_tiled`                     | `gemm_tiled_padding`             | 首轮观察问题                             |
| ----------------------- | ------------------------------------------ | -------------------------------- | -------------------------------- | ---------------------------------------- |
| Time / Duration         | 页面顶部或 `GPU Speed Of Light Throughput` | 2.93 ms                          | 4.39 ms                          | padding 是否稳定更快                     |
| Compute (SM) Throughput | `GPU Speed Of Light Throughput`            | 96.37%                           | 96.80%                           | SM 繁忙程度是否明显变化                  |
| Memory Throughput       | `GPU Speed Of Light Throughput`            | 96.37%                           | 96.80%                           | memory 汇总压力是否变化                  |
| DRAM Throughput         | `GPU Speed Of Light Throughput`            | 1.33%                            | 0.95%                            | 变化是否来自片外显存                     |
| L2 Cache Throughput     | `GPU Speed Of Light Throughput`            | 32.95%                           | 22.01%                           | L2 路径是否基本一致                      |
| Achieved Occupancy      | `Occupancy`                                | 98.21%                           | 97.35%                           | 增加 128 B shared 是否影响实际 occupancy |
| Registers / Thread      | `Launch Statistics`                        | 38                               | 36                               | 代码结构相同，寄存器是否保持一致         |
| Top Stall Reason        | `Warp State Statistics`                    | MIO Throttle：20.4 cycles，51.5% | MIO Throttle：28.5 cycles，58.6% | 名称、cycles 和占比是否变化              |

`Top Stall Reason` 请使用下面的完整格式填写：

```text
名称：
cycles per issued instruction：
占 Warp Cycles Per Issued Instruction 的比例：
NCU Estimated Speedup：
```

## 八指标填写后的第一轮判断

### 客观变化

```text
Duration：          2.93 -> 4.39 ms，增加 49.8%，padding 约慢 1.50x
Compute：           96.37% -> 96.80%，增加 0.43 个百分点
Memory：            96.37% -> 96.80%，增加 0.43 个百分点
DRAM：               1.33% -> 0.95%，下降 0.38 个百分点
L2：                32.95% -> 22.01%，下降 10.94 个百分点
Occupancy：         98.21% -> 97.35%，下降 0.86 个百分点
Registers / Thread：38 -> 36，减少 2 个
Top Stall：         MIO Throttle 仍是第一位
                    20.4 -> 28.5 cycles，增加 39.7%
                    51.5% -> 58.6%，增加 7.1 个百分点
```

### 其他七项能排除什么

- `Compute` 和 `Memory Throughput` 都只变化 0.43 个百分点，但 Duration 增加近 50%。这两个汇总利用率没有显示“做了更多有效计算”，只能说明最繁忙的 SM/片上路径仍接近饱和。
- `DRAM Throughput` 始终低于 2%，而且 padding 版本更低，因此退化不符合 DRAM 带宽跑满的特征。
- 两个 kernel 的 global-memory 地址和 global load 数量没有改变。`L2 Throughput` 从 32.95% 降到 22.01% 更像是执行时间拉长后单位时间请求率下降，不支持“L2 压力增加导致变慢”。还需查看 L2 bytes/requests 才能比较绝对工作量。
- `Achieved Occupancy` 只下降 0.86 个百分点，仍在 97% 以上；增加的 128 B shared memory 没有造成明显 occupancy 限制。
- `Registers / Thread` 反而从 38 降到 36，因此退化不能归因于寄存器增加或寄存器限制 occupancy。
- 最强的相关证据是 `MIO Throttle`：等待增加 8.1 cycles，增幅约 39.7%，占比也从 51.5% 增至 58.6%。这与 padding 版本在 shared/MIO 路径上承受更大压力相符。

### 是否符合代码修改的预期

如果预期是“经典 `+1 padding` 一定消除 bank conflict 并加速”，当前结果**不符合 H1**。

但根据这个 kernel 的实际 block 排布，Duration 增加**符合 H2，也符合地址映射能够给出的预期**。一个 warp 包含两行线程：前 16 个线程的 `threadIdx.y=0`，后 16 个线程的 `threadIdx.y=1`。以 shared store `As[ty][tx]` 为例：

```text
[16][16]，每行步长 16 个 float：
第 0 行 tx=0..15 -> bank 0..15
第 1 行 tx=0..15 -> bank 16..31
两行没有使用同一个 bank

[16][17]，每行步长 17 个 float：
第 0 行 tx=0..15 -> bank 0..15
第 1 行 tx=0..15 -> bank 17..31, 0
bank 0 同时收到两个不同地址，可能形成额外的 2-way store conflict
```

计算阶段的 `As[ty][kk]` 和 `Bs[kk][tx]` 包含同地址广播，原始 `[16][16]` 布局并没有表现出经典转置访问中的 32-way conflict。因此，盲目添加 `+1 padding` 不但可能没有可消除的冲突，还可能破坏原来的 bank 映射。

### 第一轮结论

八个指标已经支持下面的中间结论：

```text
padding 版本确实明显变慢
  -> 不是 DRAM 带宽、寄存器增加或 occupancy 明显下降造成
  -> MIO 等待显著增加
  -> 与 [16][17] shared 布局增加片上访问压力的代码预期一致
```

但这仍只达到 **证据等级 L2（定位）**，还不是 bank conflict 的最终因果证明。下一步必须填写 Shared Load/Store Bank Conflicts、Wavefronts / Requests 和 Excessive Wavefronts：

- 如果 padding 的 store conflict 和 store wavefront 明显增加，就能直接支持上面的 bank 映射推理。
- 如果 conflict 没有增加，则需要回到 Source/SASS，检查编译后的 shared 指令、MIO pipeline 和其他指令差异。

## 第二轮：Shared Memory 补充指标

填写完八指标后，在 `Memory Workload Analysis` 标题栏右上角切换视图：

```text
Details
  -> Memory Workload Analysis
  -> 点击右上角 Memory Chart 下拉菜单
  -> 选择 Memory Tables
  -> 展开 Shared Memory
```

不要在当前 Memory Chart 的节点图中寻找独立的 Shared Memory 表。Memory Chart 和 Memory Tables 是同一组分析数据的两种视图。不同 NCU 版本的下拉选项可能显示为 `Memory Tables`、`Memory Workload Analysis Tables`，或者直接列出 `Shared Memory`。

Shared Memory 表通常包含 `Instructions`、`Requests`、`Wavefronts`、`% Peak` 和 `Bank Conflicts`，行中区分 Shared Load 与 Shared Store。

| Shared-memory 指标             | `gemm_tiled` | `gemm_tiled_padding` | 从哪里得到                         |
| ------------------------------ | ------------ | -------------------- | ---------------------------------- |
| Shared Load Requests           | 335,544,320  | 536,870,912          | Shared Memory 表的 Load 行         |
| Shared Load Wavefronts         | 402,688,509  | 536,888,686          | Shared Memory 表的 Load 行         |
| Shared Load Bank Conflicts     | 0            | 0                    | Shared Memory 表的 Load 行         |
| Load Wavefronts / Requests     | 1.20011      | 1.00003              | 两列分别使用 Wavefronts / Requests |
| Shared Store Requests          | 33,554,432   | 33,554,432           | Shared Memory 表的 Store 行        |
| Shared Store Wavefronts        | 33,554,432   | 67,108,864           | Shared Memory 表的 Store 行        |
| Shared Store Bank Conflicts    | 0            | 33,554,432           | Shared Memory 表的 Store 行        |
| Store Wavefronts / Requests    | 1.0          | 2.0                  | 两列分别使用 Wavefronts / Requests |
| Store Conflict Wavefront Share | 0%           | 50.00%               | Bank Conflicts / Wavefronts        |
| Total Shared Requests          | 369,098,752  | 570,425,344          | Shared Memory 表的 Total 行        |
| Total Shared Wavefronts        | 440,699,670  | 608,454,300          | Shared Memory 表的 Total 行        |
| Total Shared Bank Conflicts    | 0            | 33,554,432           | Shared Memory 表的 Total 行        |

## 如何根据数据判断 bank conflict

### 先建立 Request -> Wavefront 的硬件框架

一个 warp 执行一条 shared-memory 指令时，可以按下面的过程理解：

```text
一条 warp shared-memory 指令
  -> 生成一个 Request：这 32 个线程想访问哪些 shared 地址
  -> 硬件检查这些地址分别落到哪个 bank
  -> 将 Request 拆成一个或多个无冲突的 Wavefront
  -> 不同 Wavefront 在不同周期串行处理
```

可以把 32 个 bank 想成 32 个并行服务窗口，把一个 wavefront 想成“一轮可以同时完成的访问”：

- 32 个线程访问 32 个不同 bank：一轮完成。
- 两个线程访问同一 bank 中的两个不同地址：同一轮无法完成，需要拆成两轮。
- 多个线程读取同一个 bank 中的同一个 32-bit 地址：硬件可以 broadcast，通常不构成 conflict。
- 每线程访问的数据宽度大于 32 bit 时，即使没有 conflict，也可能天然需要多个 wavefront。

因此需要区分三个数量：

```text
Requests：          源程序实际发出了多少次 warp memory request
Ideal Wavefronts：  考虑访问宽度和参与线程后，无 bank conflict 时本来需要多少 wavefront
Actual Wavefronts： 硬件实际执行了多少 wavefront
```

bank conflict 的概念关系是：

```text
额外 conflict wavefronts
= Actual Wavefronts - Ideal Wavefronts
```

NCU 的 `Bank Conflicts` 列就是判断“实际值是否超过理想值”的直接结果。

只有当访问是简单的 32-bit scalar load/store，并且所有参与地址在理想情况下能一轮处理时，才有：

```text
Ideal Wavefronts = Requests

平均 conflict degree
= Actual Wavefronts / Requests
```

padding 的 shared store 正好满足这个简单条件，所以可以用 Requests 与 Wavefronts 直接交叉验证。这个公式不能不加判断地套到所有 shared load/store。

判断顺序固定为：

```text
1. 先看 Bank Conflicts 是否大于 0
2. 再判断无冲突时应该需要多少 Ideal Wavefronts
3. 对比 Actual Wavefronts 是否超过理想值
4. 简单 32-bit scalar 访问可用 Requests 近似 Ideal Wavefronts
5. 最后用源码地址映射解释为什么会冲突
```

### 1. `gemm_tiled` 为什么判断为没有 conflict

```text
Shared Load Bank Conflicts  = 0
Shared Store Bank Conflicts = 0
```

NCU 对 load 和 store 都直接报告 `0`，因此当前报告不支持 shared-memory bank conflict。

特别是 store：

```text
Requests                  = 33,554,432
Wavefronts                = 33,554,432
Wavefronts / Requests     = 1.0
```

每个 store request 只需要一个 wavefront，没有因为 bank 冲突被拆成额外的串行 wavefront。

### 2. `gemm_tiled_padding` 为什么判断为 2-way store conflict

padding 的 shared load 仍然没有冲突：

```text
Shared Load Bank Conflicts = 0
```

但 shared store 出现：

```text
Requests                  = 33,554,432
Wavefronts                = 67,108,864
Bank Conflicts            = 33,554,432
Wavefronts / Requests     = 2.0
```

三个数据能够互相验证：

```text
67,108,864 / 33,554,432 = 2.0

多出的 Wavefronts
= 67,108,864 - 33,554,432
= 33,554,432
= NCU 报告的 Bank Conflicts
```

也就是说，每个 shared store request 原本一个 wavefront 可以完成，现在平均需要两个 wavefront 串行处理，因此 NCU 将它报告为平均 `2.0-way bank conflict`。

`33,554,432 / 67,108,864 = 50%` 表示额外 conflict wavefront 占全部 store wavefront 的一半。

### 3. 为什么不能只看 `Wavefronts / Requests`

`gemm_tiled` 的 shared load 数据是：

```text
Wavefronts / Requests = 1.20011
Bank Conflicts        = 0
```

这说明比例大于 1 不一定来自 bank conflict。Wavefront 数还会受到访问宽度、参与线程和编译后指令组合影响；某些访问在无冲突时的 `Ideal Wavefronts` 本来就可能大于 Requests。因此判断时：

```text
Bank Conflicts 列                 = 是否存在冲突的直接证据
Actual / Ideal Wavefronts         = 最通用的冲突交叉验证
Wavefronts / Requests             = 简单 scalar 访问下的辅助验证
NCU 给出的 N-way                 = 冲突严重程度
```

只有 `Bank Conflicts > 0`，并且实际 wavefront 相比无冲突情况增加时，才能把额外串行工作归因于 bank conflict。

### 4. 数据如何对应代码

对于 4-byte `float`，可以用下面的简化公式分析 bank：

```text
bank = shared memory 中的 float 下标 % 32
```

一个 `16x16` block 的 warp 会跨两行。执行 `As[ty][tx]` 或 `Bs[ty][tx]` store 时：

```text
[16][16]：
第 0 行 -> bank 0..15
第 1 行 -> bank 16..31
没有两个不同地址落到同一个 bank

[16][17]：
第 0 行 -> bank 0..15
第 1 行 -> bank 17..31, 0
bank 0 上出现两个不同地址，需要拆成两个 wavefront
```

因此源码预测和 NCU 数据形成了完整证据链：

```text
[16][17] 改变行步长
  -> shared store 的两个不同地址落到 bank 0
  -> Store Bank Conflicts = 33,554,432
  -> Store Wavefronts / Requests = 2.0
  -> MIO Throttle 增加
  -> Duration 增加
```

读取这张表时注意：

```text
Wavefronts / Requests：描述平均每个 request 需要多少个 wavefront
Bank Conflicts：NCU 对 bank conflict 的直接统计
```

`Wavefronts / Requests > 1` 不等于一定存在 bank conflict。例如 tiled 的 shared load 比例是 `1.20011`，但 NCU 直接报告 `0` 个 load bank conflict。判断冲突时应以 `Bank Conflicts` 和 NCU 给出的 N-way conflict 为主，比例用于说明实际处理工作量。

当前 padding 截图中的规则卡已经给出：

```text
Shared Load Requests        = 536,870,912
Shared Load Wavefronts      = 536,888,686
Shared Load Bank Conflicts  = 0
Load Wavefronts / Requests  = 1.00003，约等于 1

Shared Store Requests       = 33,554,432
Shared Store Wavefronts     = 67,108,864
Shared Store Bank Conflicts = 33,554,432
Wavefronts / Requests       = 2.0
Conflict Wavefront Share    = 50.00%
NCU Estimated Speedup       = 48.98%
```

这说明 shared load 基本是一次 request 对应一个 wavefront，并且 NCU 直接报告 `0` 个 load bank conflict。问题集中在 shared store：每个 store request 平均需要两个 wavefront，padding 版本确实出现了 2-way shared-store bank conflict。

对应的 tiled 截图给出：

```text
Shared Load Requests        = 335,544,320
Shared Load Wavefronts      = 402,688,509
Shared Load Bank Conflicts  = 0
Load Wavefronts / Requests  = 1.20011

Shared Store Requests       = 33,554,432
Shared Store Wavefronts     = 33,554,432
Shared Store Bank Conflicts = 0
Store Wavefronts / Requests = 1.0
```

两列对比后可以看到：padding 不仅新增了 shared-store bank conflict，shared load requests 也从 `335,544,320` 增加到 `536,870,912`，增幅为 60%。因此 MIO 压力增加不应只归因于 store conflict，还包括编译后 shared-load 指令/request 数量增加；下一步应在 Source/SASS 中比较两版的 LDS 指令。

如果右上角下拉菜单中没有 Memory Tables，可以直接从现有报告导出 shared 指标：

```bash
ncu --import reports/ncu/gemm_tiled_padding_full.ncu-rep \
  --page raw \
  --kernel-name-base function \
  --kernel-id "::gemm_tiled_padding_kernel:6" \
  --metrics group:memory__shared_table
```

若当前 NCU 版本不接受 metric group，先查询实际指标名称：

```bash
ncu --query-metrics-mode all \
  | grep -E 'bank_conflicts.*shared|wavefronts.*shared|requests.*shared'
```

## 4090 上的执行命令

如果报告尚未生成：

```bash
ITERS=1 bash scripts/profile_ncu_full.sh gemm_tiled
ITERS=1 bash scripts/profile_ncu_full.sh gemm_tiled_padding
```

快速提取八个指标：

```bash
bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_tiled_full.ncu-rep \
  gemm_tiled_kernel \
  6

bash scripts/extract_ncu_metrics.sh \
  reports/ncu/gemm_tiled_padding_full.ncu-rep \
  gemm_tiled_padding_kernel \
  6
```

脚本提取结果用于快速填写，Shared Memory 表仍需单独检查。

## 最终结论模板

完成两轮填写后再写结论：

```text
1. 正确性是否一致：

2. benchmark 中位 Duration：
   tiled：
   tiled_padding：
   波动范围：
   差异是否超过测试波动：

3. 八指标中的主要变化：

4. bank conflict / wavefront 的变化：

5. 代码到指标的因果链：
   [16][16] -> [16][17]
     -> shared 地址步长变化
     -> bank conflict / wavefront：
     -> MIO 或 Short Scoreboard：
     -> Duration：

6. 当前结论支持 H1 还是 H2：

7. 仍不能确认的问题：
```

只有同时看到正确性一致、Duration 的稳定变化，以及 bank-conflict 相关指标符合预期，才能认为 padding 对当前访问模式产生了可验证的性能影响。

## 相关文档

- [第一组：gemm_naive vs gemm_tiled](gemm_naive_vs_tiled.md)
- [NCU 八个关键指标详解](ncu_eight_key_metrics.md)
- [NCU Warp Stall 原因与解决思路](ncu_warp_stall_reasons.md)
- [NCU 从 GEMM 演进到陌生算子的分析工作流](ncu_unknown_kernel_analysis_workflow.md)
