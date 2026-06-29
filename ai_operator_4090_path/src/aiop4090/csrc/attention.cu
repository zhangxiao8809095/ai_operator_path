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

struct AttentionOnlineState {
    float max_val;
    float sum_val;
    float acc_val;
};

__forceinline__ __device__ AttentionOnlineState combine_attention_online(AttentionOnlineState a,
                                                                        AttentionOnlineState b) {
    if (a.sum_val == 0.0f) return b;
    if (b.sum_val == 0.0f) return a;
    float max_val = fmaxf(a.max_val, b.max_val);
    float a_scale = expf(a.max_val - max_val);
    float b_scale = expf(b.max_val - max_val);
    return {
        max_val,
        a.sum_val * a_scale + b.sum_val * b_scale,
        a.acc_val * a_scale + b.acc_val * b_scale,
    };
}

// Q shape: [B,H,1,D]. K/V cache shape: [B,H,S,D]. O shape: [B,H,1,D].
// Grid: x=D(out feature), y=B*H. Each block decodes one O[b,h,0,d_out].
__global__ void attention_kv_cache_decode_kernel(const float* __restrict__ Q,
                                                 const float* __restrict__ K,
                                                 const float* __restrict__ V,
                                                 float* __restrict__ O,
                                                 int B, int H, int S, int D,
                                                 int kv_len) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem;
    float* sacc = smem;

    int d_out = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    int q_base = ((b * H + h) * 1) * D;
    int kv_base = ((b * H + h) * S);
    float scale = rsqrtf(static_cast<float>(D));

    float local_max = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float score = 0.0f;
        int k_base = (kv_base + j) * D;
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
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float score = 0.0f;
        int k_base = (kv_base + j) * D;
        for (int d = 0; d < D; ++d) {
            score += Q[q_base + d] * K[k_base + d];
        }
        score *= scale;
        float p = expf(score - max_val);
        local_sum += p;
        local_acc += p * V[(kv_base + j) * D + d_out];
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
        O[((b * H + h) * 1) * D + d_out] = sacc[0] / denom;
    }
}

// Same output contract as attention_naive_kernel, but computes softmax denominator
// and weighted V in one online pass over tiled key positions.
__global__ void attention_tiled_online_softmax_kernel(const float* __restrict__ Q,
                                                      const float* __restrict__ K,
                                                      const float* __restrict__ V,
                                                      float* __restrict__ O,
                                                      int B, int H, int S, int D,
                                                      bool causal) {
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;
    float* smem_acc = smem + 2 * blockDim.x;

    int d_out = blockIdx.x;
    int i = blockIdx.y;
    int bh = blockIdx.z;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    int base = ((b * H + h) * S);
    int q_base = (base + i) * D;
    float scale = rsqrtf(static_cast<float>(D));

    AttentionOnlineState local{-FLT_MAX, 0.0f, 0.0f};
    constexpr int TILE_N = 128;
    int kv_limit = causal ? (i + 1) : S;
    for (int tile = 0; tile < kv_limit; tile += TILE_N) {
        int tile_end = tile + TILE_N;
        if (tile_end > kv_limit) tile_end = kv_limit;
        for (int j = tile + tid; j < tile_end; j += blockDim.x) {
            float score = 0.0f;
            int k_base = (base + j) * D;
            for (int d = 0; d < D; ++d) {
                score += Q[q_base + d] * K[k_base + d];
            }
            score *= scale;
            AttentionOnlineState item{score, 1.0f, V[(base + j) * D + d_out]};
            local = combine_attention_online(local, item);
        }
    }

    smem_max[tid] = local.max_val;
    smem_sum[tid] = local.sum_val;
    smem_acc[tid] = local.acc_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            AttentionOnlineState a{smem_max[tid], smem_sum[tid], smem_acc[tid]};
            AttentionOnlineState b{smem_max[tid + stride], smem_sum[tid + stride], smem_acc[tid + stride]};
            AttentionOnlineState combined = combine_attention_online(a, b);
            smem_max[tid] = combined.max_val;
            smem_sum[tid] = combined.sum_val;
            smem_acc[tid] = combined.acc_val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        O[(base + i) * D + d_out] = smem_acc[0] / smem_sum[0];
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

torch::Tensor attention_causal_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
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
        B, H, S, D, true);
    return O;
}

torch::Tensor attention_kv_cache_decode(torch::Tensor Q,
                                        torch::Tensor K_cache,
                                        torch::Tensor V_cache,
                                        int kv_len) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K_cache);
    CHECK_INPUT(V_cache);
    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,1,D]");
    TORCH_CHECK(K_cache.dim() == 4 && V_cache.dim() == 4, "K/V cache must be [B,H,S,D]");
    TORCH_CHECK(Q.size(2) == 1, "Q decode length must be 1");
    TORCH_CHECK(K_cache.sizes() == V_cache.sizes(), "K/V cache shape mismatch");
    TORCH_CHECK(Q.size(0) == K_cache.size(0) &&
                Q.size(1) == K_cache.size(1) &&
                Q.size(3) == K_cache.size(3),
                "Q and K/V cache B/H/D mismatch");
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(K_cache.size(2));
    int D = static_cast<int>(Q.size(3));
    TORCH_CHECK(kv_len > 0 && kv_len <= S, "kv_len must be in [1, cache sequence length]");
    auto O = torch::empty_like(Q);
    int block = 128;
    dim3 grid(D, B * H);
    attention_kv_cache_decode_kernel<<<grid, block, block * sizeof(float)>>>(
        Q.data_ptr<float>(), K_cache.data_ptr<float>(), V_cache.data_ptr<float>(), O.data_ptr<float>(),
        B, H, S, D, kv_len);
    return O;
}

torch::Tensor attention_tiled_online_softmax(torch::Tensor Q,
                                             torch::Tensor K,
                                             torch::Tensor V,
                                             bool causal) {
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
    attention_tiled_online_softmax_kernel<<<grid, block, 3 * block * sizeof(float)>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        B, H, S, D, causal);
    return O;
}
