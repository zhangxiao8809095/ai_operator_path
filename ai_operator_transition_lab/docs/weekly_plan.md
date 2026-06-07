# 8-Week Operator Portfolio Plan

## Week 1: Environment + GEMM baseline
- GPU: RTX 4090/5090/A10/L4 single card
- Code: gemm_naive, benchmark, correctness test
- Profile: ncu GPU Speed of Light, Memory Workload Analysis
- Deliverable: naive GEMM report

## Week 2: GEMM tiled + bank conflict
- Code: gemm_tiled with shared memory and padding
- Profile: DRAM throughput, L1/TEX, shared memory behavior
- Deliverable: naive vs tiled report

## Week 3: Softmax
- Code: row-wise max trick softmax
- Profile: memory throughput, occupancy, launch overhead
- Deliverable: softmax numerical stability + profiling report

## Week 4: LayerNorm / RMSNorm
- Code: layernorm_row, rmsnorm_row
- Profile: reduction cost, memory bandwidth, register pressure
- Deliverable: LayerNorm vs RMSNorm report

## Week 5: PyTorch extension polish
- Code: binding.cpp, setup.py, tests
- Deliverable: one-command install/test/benchmark workflow

## Week 6: Naive causal attention
- Code: attention_naive, causal mask, torch comparison
- Profile: S^2 score cost and HBM behavior
- Deliverable: why naive attention is slow report

## Week 7: Online softmax + mini FlashAttention explanation
- Code: docs/online_softmax_reference.py and attention profiling
- Deliverable: online softmax / FlashAttention IO-saving explanation

## Week 8: Resume and interview packaging
- Code: README cleanup, benchmark table, profiling reports
- Deliverable: resume project bullets and interview Q&A
