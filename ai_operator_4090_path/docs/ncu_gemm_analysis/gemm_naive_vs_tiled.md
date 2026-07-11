# NCU GEMM 入门：`gemm_naive` vs `gemm_tiled`

这份文档只做第一轮对比，不追求一次看懂 NCU 的全部指标。

本轮只回答两个问题：

1. `gemm_tiled` 是否比 `gemm_naive` 更快？
2. 使用 shared memory 后，主要瓶颈发生了什么变化？

八个指标的定义、判断方法和常见误区见：[NCU 八个关键指标详解](ncu_eight_key_metrics.md)。

## 代码差异

| Kernel | 主要做法 |
| --- | --- |
| `gemm_naive` | 每个线程计算一个输出元素，并在循环中反复从 global memory 读取 `A` 和 `B`。 |
| `gemm_tiled` | 线程协作把 `A` 和 `B` 的小块搬到 shared memory，再在 block 内重复使用。 |

对照源码：[gemm.cu](../../src/aiop4090/csrc/gemm.cu)

当前预期：`gemm_tiled` 通过数据复用减少 global memory 压力，但会增加 shared memory 访问和 `__syncthreads()`。

## 关键指标记录与位置

在 NCU 中分别选择一条稳定的 `gemm_naive_kernel` 和 `gemm_tiled_kernel` 记录，然后按照每一行的位置查找并填写。暂时只看这 8 项。

| 指标                                      | NCU `Details` 中的位置                  | `gemm_naive` | `gemm_tiled` | 重点观察                         |
| ----------------------------------------- | --------------------------------------- | ------------: | ------------: | -------------------------------- |
| Time / Duration                           | 页面顶部或 `GPU Speed Of Light Throughput` | 3.82 ms      | 2.93 ms      | tiled 是否更快                   |
| SM utilization / Compute Throughput       | `GPU Speed Of Light Throughput`         | 98.35%       | 96.38%       | 计算单元是否更忙                 |
| Memory utilization / Memory Throughput    | `GPU Speed Of Light Throughput`         | 98.35%       | 96.38%       | 是否仍有较高的访存压力           |
| DRAM Throughput                           | `GPU Speed Of Light Throughput`         | 1.09%        | 1.32%        | 显存带宽是否繁忙                 |
| L2 Throughput                             | `GPU Speed Of Light Throughput`         | 37.83%       | 33.03%       | L2 cache 层流量是否变化          |
| Achieved Occupancy                        | `Occupancy`                             | 98.09%       | 98.20%       | 实际并行度是否变化               |
| Registers / Thread                        | `Launch Statistics`                     | 40           | 38           | tiled 是否使用更多寄存器         |
| Top Stall Reason                          | `Warp State Statistics`                 | LG Throttle：27.0 cycles，65.6% | MIO Throttle：20.4 cycles，51.5% | warp 主要在等待什么              |

不同 NCU 版本的名称可能略有差异。先记录界面显示的原始名称和数值，不急着解释所有子指标。

## 实际代码改动

### `gemm_naive`

每个线程计算一个 `C[row, col]`。在 `K` 次循环中，每次都直接从 global memory 读取一个 `A` 和一个 `B`：

```cpp
for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
}
```

因此，每个线程大约发出 `2K` 次 global load。`K=2048` 时约为 `4096` 次。

### `gemm_tiled`

`tiled` 每次让整个 block 合作读取一个 `16x16` 的 `A/B` 小块：

```cpp
As[ty][tx] = A[...];
Bs[ty][tx] = B[...];
__syncthreads();

for (int kk = 0; kk < 16; ++kk) {
    acc += As[ty][kk] * Bs[kk][tx];
}
__syncthreads();
```

一个 global memory 元素被搬进 shared memory 后，可以被 block 内多个线程复用：一个 `A` 元素供同一行的 16 个线程使用，一个 `B` 元素供同一列的 16 个线程使用。因此，每个线程的 global load 数量从约 `2K` 降到 `2K/16`；`K=2048` 时从约 `4096` 次降到 `256` 次。

代价是新增 shared-memory 读写，并且每处理一个 tile 都需要两次 `__syncthreads()`。

## 代码改动如何影响指标

下表固定按“同一行看一个指标、同一列看一个 kernel”的方式排列：

| 指标                   | `gemm_naive`                         | `gemm_tiled`                         | 代码改动 → 硬件行为 → 指标结果 |
| ---------------------- | ------------------------------------ | ------------------------------------ | ------------------------------ |
| Time / Duration        | `3.82 ms`                            | `2.93 ms`                            | global load 约减少 16 倍，但新增 shared-memory 访问和同步 → 时间下降 23.3%，加速约 `1.30x` |
| Compute Throughput     | `98.35%`                             | `96.38%`                             | 数学计算量不变，shared-memory 和同步改变了指令构成 → 两者都很高，tiled 低 1.97 个百分点 |
| Memory Throughput      | `98.35%`                             | `96.38%`                             | 压力从 global/LG 路径转向 shared/MIO 路径 → 总利用率仍高，但繁忙路径发生转移 |
| DRAM Throughput        | `1.09%`                              | `1.32%`                              | 两者都没有跑满显存带宽；tiled 时间更短会使单位时间吞吐率略升 → 不能据此认为 global load 增多 |
| L2 Throughput          | `37.83%`                             | `33.03%`                             | block 内使用 shared memory 复用数据 → 重复访问 cache 的压力下降 4.80 个百分点 |
| Achieved Occupancy     | `98.09%`                             | `98.20%`                             | 都是 256 threads/block，tiled 只增加 `2048 B` shared memory 且寄存器未增加 → occupancy 基本不变 |
| Registers / Thread     | `40`                                 | `38`                                 | shared 数组不占线程寄存器；编译后的变量生命周期发生变化 → 减少 2 个，未影响 occupancy |
| Top Stall Reason       | `LG Throttle (27.0 cycles, 65.6%)`  | `MIO Throttle (20.4 cycles, 51.5%)` | 内层循环反复读取 global memory 改为反复读取 shared memory → 主要等待从 LG 队列转移到 MIO 队列 |

阅读顺序：先沿 `gemm_naive` 和 `gemm_tiled` 两列纵向比较数值，再横向查看最后一列的因果解释。

## 第一轮结论

```text
1. gemm_tiled 的时间从 3.82 ms 降到 2.93 ms，比 gemm_naive 快约 1.30 倍。
2. DRAM 始终很低，L2 从 37.83% 降到 33.03%；总 Memory Throughput 仍然很高，说明压力主要在片上 memory 路径，而不是显存带宽。
3. naive 的主要等待是 LG Throttle，tiled 的主要等待是 MIO Throttle；瓶颈从频繁发射 global-memory 指令，转移到了 shared-memory 指令流水线。
```

目前不能只根据这 8 项判断 shared-memory bank conflict 或同步开销的具体大小；它们需要在下一轮实验中单独验证。
