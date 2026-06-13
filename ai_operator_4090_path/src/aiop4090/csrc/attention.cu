#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

// Q/K/V/O shape: [B, H, S, D], contiguous, fp32.
// Grid: x=D(out feature), y=S(query position), z=B*H.
// Each block computes one O[b,h,i,d_out]. Threads reduce over key positions j.
// This is intentionally simple and slow. It is for correctness + Nsight observation,
// then you replace it with tiled/online-softmax versions during the mini-FlashAttention phase.
__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int B, int H, int S, int D,
                                       bool causal) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem;
    float* sacc = smem;

    int d_out = blockIdx.x;
    int i = blockIdx.y;
    int bh = blockIdx.z;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    int base = ((b * H + h) * S);
    float scale = rsqrtf(static_cast<float>(D));

    float local_max = -FLT_MAX;
    for (int j = tid; j < S; j += blockDim.x) {
        if (causal && j > i) continue;
        float score = 0.0f;
        int q_base = (base + i) * D;
        int k_base = (base + j) * D;
        for (int d = 0; d < D; ++d) {
            score += Q[q_base + d] * K[k_base + d];
        }
        score *= scale;
        local_max = fmaxf(local_max, score);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float max_val = smax[0];

    float local_sum = 0.0f;
    float local_acc = 0.0f;
    for (int j = tid; j < S; j += blockDim.x) {
        if (causal && j > i) continue;
        float score = 0.0f;
        int q_base = (base + i) * D;
        int k_base = (base + j) * D;
        for (int d = 0; d < D; ++d) {
            score += Q[q_base + d] * K[k_base + d];
        }
        score *= scale;
        float p = expf(score - max_val);
        local_sum += p;
        local_acc += p * V[(base + j) * D + d_out];
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float denom = ssum[0];

    sacc[tid] = local_acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sacc[tid] += sacc[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        O[(base + i) * D + d_out] = sacc[0] / denom;
    }
}

} // namespace

torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q/K/V must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V shape mismatch");
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));
    auto O = torch::empty_like(Q);
    int block = 128;
    dim3 grid(D, S, B * H);
    attention_naive_kernel<<<grid, block, block * sizeof(float)>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        B, H, S, D, causal);
    return O;
}
