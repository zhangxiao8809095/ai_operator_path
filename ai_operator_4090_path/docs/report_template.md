# Kernel Profiling Report Template

## 1. Kernel 信息

- Kernel 名称：
- 输入 shape：
- GPU：RTX 4090
- dtype：fp32
- 版本：naive / tiled / regtile / online / fused

## 2. 正确性验证

- 对拍对象：PyTorch baseline
- atol / rtol：
- max absolute error：
- 是否通过：

## 3. Benchmark

| Version | Shape | Latency ms | TFLOP/s or GB/s | Speedup |
|---|---:|---:|---:|---:|
| baseline | | | | |
| current | | | | |

## 4. Nsight Compute 关键指标

| Metric | Value | 判断 |
|---|---:|---|
| Duration | | |
| SM Throughput | | compute 是否吃满 |
| Memory Throughput | | memory 是否吃满 |
| DRAM Throughput | | 是否受 HBM/GDDR 约束 |
| L1/TEX Throughput | | cache/shared 访问情况 |
| L2 Throughput | | L2 压力 |
| Achieved Occupancy | | occupancy 是否受限 |
| Registers / Thread | | 寄存器压力 |
| Shared Memory / Block | | shared memory 压力 |
| Warp Stall Top Reasons | | 主要停顿原因 |

## 5. 瓶颈判断

结论：compute-bound / memory-bound / latency-bound / launch-overhead-bound

依据：
1. 
2. 
3. 

## 6. 优化动作

当前版本做了什么：
- 
- 

指标变化是否符合预期：
- 

## 7. 下一步优化

下一步准备做：
- 
- 
