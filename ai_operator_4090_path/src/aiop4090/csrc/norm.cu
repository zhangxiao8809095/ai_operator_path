#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__forceinline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__forceinline__ __device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < warp_count) ? smem[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

bool is_aligned_16(const torch::Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr<float>()) % 16) == 0;
}

__global__ void layernorm_row_kernel(const float* __restrict__ X,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ Y,
                                     int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += X[row * cols + col];
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float mean = smem[0] / cols;

    float local_var = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = X[row * cols + col] - mean;
        local_var += v * v;
    }
    smem[tid] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / cols + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        float norm = (X[row * cols + col] - mean) * inv_std;
        Y[row * cols + col] = norm * gamma[col] + beta[col];
    }
}

__global__ void rmsnorm_row_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ Y,
                                   int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = X[row * cols + col];
        local_sum_sq += v * v;
    }
    smem[tid] = local_sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / cols + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];
    }
}

__global__ void layernorm_warp_reduce_kernel(const float* __restrict__ X,
                                             const float* __restrict__ gamma,
                                             const float* __restrict__ beta,
                                             float* __restrict__ Y,
                                             int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += X[row * cols + col];
    }
    float mean = block_reduce_sum(local_sum, smem) / cols;

    float local_var = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = X[row * cols + col] - mean;
        local_var += v * v;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, smem) / cols + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        float norm = (X[row * cols + col] - mean) * inv_std;
        Y[row * cols + col] = norm * gamma[col] + beta[col];
    }
}

__global__ void rmsnorm_warp_reduce_kernel(const float* __restrict__ X,
                                           const float* __restrict__ gamma,
                                           float* __restrict__ Y,
                                           int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = X[row * cols + col];
        local_sum_sq += v * v;
    }
    float inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];
    }
}

__global__ void layernorm_vectorized_kernel(const float* __restrict__ X,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            float* __restrict__ Y,
                                            int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int vec_cols = cols / 4;

    const float4* X4 = reinterpret_cast<const float4*>(X + row * cols);
    float local_sum = 0.0f;
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        local_sum += x.x + x.y + x.z + x.w;
    }
    float mean = block_reduce_sum(local_sum, smem) / cols;

    float local_var = 0.0f;
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        float v0 = x.x - mean;
        float v1 = x.y - mean;
        float v2 = x.z - mean;
        float v3 = x.w - mean;
        local_var += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, smem) / cols + eps);

    float4* Y4 = reinterpret_cast<float4*>(Y + row * cols);
    const float4* G4 = reinterpret_cast<const float4*>(gamma);
    const float4* B4 = reinterpret_cast<const float4*>(beta);
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        float4 g = G4[vec_col];
        float4 b = B4[vec_col];
        float4 y;
        y.x = (x.x - mean) * inv_std * g.x + b.x;
        y.y = (x.y - mean) * inv_std * g.y + b.y;
        y.z = (x.z - mean) * inv_std * g.z + b.z;
        y.w = (x.w - mean) * inv_std * g.w + b.w;
        Y4[vec_col] = y;
    }
}

__global__ void rmsnorm_vectorized_kernel(const float* __restrict__ X,
                                          const float* __restrict__ gamma,
                                          float* __restrict__ Y,
                                          int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int vec_cols = cols / 4;

    const float4* X4 = reinterpret_cast<const float4*>(X + row * cols);
    float local_sum_sq = 0.0f;
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        local_sum_sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
    float inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps);

    float4* Y4 = reinterpret_cast<float4*>(Y + row * cols);
    const float4* G4 = reinterpret_cast<const float4*>(gamma);
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        float4 g = G4[vec_col];
        float4 y;
        y.x = x.x * inv_rms * g.x;
        y.y = x.y * inv_rms * g.y;
        y.z = x.z * inv_rms * g.z;
        y.w = x.w * inv_rms * g.w;
        Y4[vec_col] = y;
    }
}

} // namespace

torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), "gamma/beta size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    int block = 256;
    layernorm_row_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    return Y;
}

torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1), "gamma size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    int block = 256;
    rmsnorm_row_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    return Y;
}

torch::Tensor layernorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), "gamma/beta size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    return Y;
}

torch::Tensor layernorm_vectorized(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), "gamma/beta size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&
                          is_aligned_16(beta) && is_aligned_16(Y);
    if (can_vectorize) {
        layernorm_vectorized_kernel<<<rows, block, block * sizeof(float)>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    } else {
        layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    }
    return Y;
}

torch::Tensor rmsnorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1), "gamma size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    return Y;
}

torch::Tensor rmsnorm_vectorized(torch::Tensor X, torch::Tensor gamma, double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1), "gamma size mismatch");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&
                          is_aligned_16(Y);
    if (can_vectorize) {
        rmsnorm_vectorized_kernel<<<rows, block, block * sizeof(float)>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    } else {
        rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    }
    return Y;
}
