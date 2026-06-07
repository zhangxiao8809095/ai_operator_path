# Kernel Profiling Report Template

## 1. Kernel
- Name:
- Shape:
- Dtype:
- GPU:

## 2. Baseline
- PyTorch/cuBLAS/cuDNN baseline latency:
- Custom kernel latency:
- Speedup or slowdown:

## 3. Expected bottleneck
- Compute-bound / memory-bound / launch-bound:
- Reason:

## 4. Nsight Compute metrics
- GPU Speed Of Light:
- SM Active:
- Compute Throughput:
- Memory Throughput:
- DRAM Throughput:
- L1/TEX Throughput:
- L2 Throughput:
- Achieved Occupancy:
- Register per thread:
- Shared memory per block:
- Top stall reason:

## 5. Interpretation
- What changed after optimization?
- Which metric improved?
- Which metric got worse?
- Is the result consistent with the theory?

## 6. Next optimization
- Code-level action:
- Expected metric change:
