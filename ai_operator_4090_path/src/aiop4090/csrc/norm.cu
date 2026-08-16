#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
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

struct WelfordState {
    float mean;
    float m2;
    float count;
};

__forceinline__ __device__ WelfordState welford_update(WelfordState state,
                                                        float value) {
    float delta = value - state.mean;
    float new_count = state.count + 1.0f;
    // setup.py enables --use_fast_math for the teaching kernels.  Force a
    // round-to-nearest reciprocal here so the statistics do not inherit the
    // approximate fast division used by ordinary `/` expressions.
    float reciprocal = __fdiv_rn(1.0f, new_count);
    float new_mean = state.mean + delta * reciprocal;
    state.m2 += delta * (value - new_mean);
    state.mean = new_mean;
    state.count = new_count;
    return state;
}

__forceinline__ __device__ WelfordState welford_combine(WelfordState a,
                                                         WelfordState b) {
    float count = a.count + b.count;
    if (count == 0.0f) return WelfordState{0.0f, 0.0f, 0.0f};

    float reciprocal = __fdiv_rn(1.0f, count);
    float weight_b = b.count * reciprocal;
    float delta = b.mean - a.mean;
    return WelfordState{
        a.mean + delta * weight_b,
        a.m2 + b.m2 + delta * delta * a.count * weight_b,
        count,
    };
}

__forceinline__ __device__ WelfordState warp_reduce_welford(WelfordState state) {
    unsigned int mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        WelfordState other{
            __shfl_down_sync(mask, state.mean, offset),
            __shfl_down_sync(mask, state.m2, offset),
            __shfl_down_sync(mask, state.count, offset),
        };
        if (lane + offset < warpSize) state = welford_combine(state, other);
    }
    return state;
}

__forceinline__ __device__ WelfordState block_reduce_welford(
        WelfordState state, WelfordState* smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    state = warp_reduce_welford(state);
    if (lane == 0) smem[warp_id] = state;
    __syncthreads();

    if (warp_id == 0) {
        state = lane < warp_count
            ? smem[lane]
            : WelfordState{0.0f, 0.0f, 0.0f};
        state = warp_reduce_welford(state);
        if (lane == 0) smem[0] = state;
    }
    __syncthreads();
    return smem[0];
}

bool is_aligned_16(const torch::Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr<float>()) % 16) == 0;
}

bool is_aligned_8(const torch::Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr<float>()) % 8) == 0;
}

__global__ void layernorm_row_kernel(const float* __restrict__ X,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ Y,
                                     int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Keep the Welford mean near zero.  Materializing the full mean again
    // would round a small residual away for inputs such as 1000 + O(1e-2).
    float shift = X[row * cols];
    WelfordState stats{0.0f, 0.0f, 0.0f};
    for (int col = 0; col < cols; ++col) {
        stats = welford_update(stats, X[row * cols + col] - shift);
    }
    float mean_delta = stats.mean;
    float variance = __fdiv_rn(stats.m2, stats.count);
    float inv_std = rsqrtf(variance + eps);
    for (int col = 0; col < cols; ++col) {
        float centered = (X[row * cols + col] - shift) - mean_delta;
        Y[row * cols + col] = centered * inv_std * gamma[col] + beta[col];
    }
}

__global__ void rmsnorm_row_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ Y,
                                   int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float sum_sq = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float v = X[row * cols + col];
        sum_sq += v * v;
    }
    float inv_rms = rsqrtf(sum_sq / cols + eps);
    for (int col = 0; col < cols; ++col) {
        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];
    }
}

__global__ void layernorm_block_reduce_kernel(const float* __restrict__ X,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float* __restrict__ Y,
                                              int rows, int cols, float eps) {
    // Dynamic shared-memory symbols with different element types must not
    // reuse the float-reduction symbol name in the same CUDA translation unit.
    extern __shared__ WelfordState welford_smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float shift = X[row * cols];
    WelfordState stats{0.0f, 0.0f, 0.0f};
    for (int col = tid; col < cols; col += blockDim.x) {
        stats = welford_update(stats, X[row * cols + col] - shift);
    }
    welford_smem[tid] = stats;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            welford_smem[tid] = welford_combine(
                welford_smem[tid], welford_smem[tid + stride]);
        }
        __syncthreads();
    }
    float mean_delta = welford_smem[0].mean;
    float variance = __fdiv_rn(welford_smem[0].m2, welford_smem[0].count);
    float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        float centered = (X[row * cols + col] - shift) - mean_delta;
        float norm = centered * inv_std;
        Y[row * cols + col] = norm * gamma[col] + beta[col];
    }
}

__global__ void rmsnorm_block_reduce_kernel(const float* __restrict__ X,
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
    extern __shared__ WelfordState welford_smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float shift = X[row * cols];
    WelfordState stats{0.0f, 0.0f, 0.0f};
    for (int col = tid; col < cols; col += blockDim.x) {
        stats = welford_update(stats, X[row * cols + col] - shift);
    }
    stats = block_reduce_welford(stats, welford_smem);
    float mean_delta = stats.mean;
    float variance = __fdiv_rn(stats.m2, stats.count);
    float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        float centered = (X[row * cols + col] - shift) - mean_delta;
        float norm = centered * inv_std;
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
    extern __shared__ WelfordState welford_smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int vec_cols = cols / 4;

    const float4* X4 = reinterpret_cast<const float4*>(X + row * cols);
    float shift = X[row * cols];
    WelfordState stats{0.0f, 0.0f, 0.0f};
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        stats = welford_update(stats, x.x - shift);
        stats = welford_update(stats, x.y - shift);
        stats = welford_update(stats, x.z - shift);
        stats = welford_update(stats, x.w - shift);
    }
    stats = block_reduce_welford(stats, welford_smem);
    float mean_delta = stats.mean;
    float variance = __fdiv_rn(stats.m2, stats.count);
    float inv_std = rsqrtf(variance + eps);

    float4* Y4 = reinterpret_cast<float4*>(Y + row * cols);
    const float4* G4 = reinterpret_cast<const float4*>(gamma);
    const float4* B4 = reinterpret_cast<const float4*>(beta);
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float4 x = X4[vec_col];
        float4 g = G4[vec_col];
        float4 b = B4[vec_col];
        float4 y;
        y.x = ((x.x - shift) - mean_delta) * inv_std * g.x + b.x;
        y.y = ((x.y - shift) - mean_delta) * inv_std * g.y + b.y;
        y.z = ((x.z - shift) - mean_delta) * inv_std * g.z + b.z;
        y.w = ((x.w - shift) - mean_delta) * inv_std * g.w + b.w;
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

__global__ void rmsnorm_vectorized_float2_kernel(const float* __restrict__ X,
                                                 const float* __restrict__ gamma,
                                                 float* __restrict__ Y,
                                                 int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int vec_cols = cols / 2;

    const float2* X2 = reinterpret_cast<const float2*>(X + row * cols);
    float local_sum_sq = 0.0f;
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float2 x = X2[vec_col];
        local_sum_sq += x.x * x.x + x.y * x.y;
    }
    float inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps);

    float2* Y2 = reinterpret_cast<float2*>(Y + row * cols);
    const float2* G2 = reinterpret_cast<const float2*>(gamma);
    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {
        float2 x = X2[vec_col];
        float2 g = G2[vec_col];
        Y2[vec_col] = make_float2(x.x * inv_rms * g.x, x.y * inv_rms * g.y);
    }
}

void check_eps(double eps) {
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and non-negative");
}

void check_layernorm_inputs(const torch::Tensor& X,
                            const torch::Tensor& gamma,
                            const torch::Tensor& beta,
                            double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_SAME_DEVICE(X, gamma);
    CHECK_SAME_DEVICE(X, beta);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), "gamma/beta size mismatch");
    CHECK_DIM_FITS_INT(X, 0);
    CHECK_DIM_FITS_INT(X, 1);
    check_eps(eps);
}

void check_rmsnorm_inputs(const torch::Tensor& X,
                          const torch::Tensor& gamma,
                          double eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_SAME_DEVICE(X, gamma);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1), "gamma size mismatch");
    CHECK_DIM_FITS_INT(X, 0);
    CHECK_DIM_FITS_INT(X, 1);
    check_eps(eps);
}

} // namespace

torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    check_layernorm_inputs(X, gamma, beta, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    int grid = ceil_div_int(rows, block);
    layernorm_row_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps) {
    check_rmsnorm_inputs(X, gamma, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    int grid = ceil_div_int(rows, block);
    rmsnorm_row_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor layernorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    check_layernorm_inputs(X, gamma, beta, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(WelfordState), at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor layernorm_block_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    check_layernorm_inputs(X, gamma, beta, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    layernorm_block_reduce_kernel<<<rows, block, block * sizeof(WelfordState), at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor layernorm_vectorized(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    check_layernorm_inputs(X, gamma, beta, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&
                          is_aligned_16(beta) && is_aligned_16(Y);
    if (can_vectorize) {
        layernorm_vectorized_kernel<<<rows, block, block * sizeof(WelfordState), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    } else {
        layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(WelfordState), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor rmsnorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, double eps) {
    check_rmsnorm_inputs(X, gamma, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor rmsnorm_block_reduce(torch::Tensor X, torch::Tensor gamma, double eps) {
    check_rmsnorm_inputs(X, gamma, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    rmsnorm_block_reduce_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor rmsnorm_vectorized(torch::Tensor X, torch::Tensor gamma, double eps) {
    check_rmsnorm_inputs(X, gamma, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    bool can_vectorize = (cols % 2 == 0) && is_aligned_8(X) && is_aligned_8(gamma) &&
                          is_aligned_8(Y);
    if (can_vectorize) {
        rmsnorm_vectorized_float2_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    } else {
        rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor rmsnorm_vectorized_float4(torch::Tensor X, torch::Tensor gamma, double eps) {
    check_rmsnorm_inputs(X, gamma, eps);
    c10::cuda::CUDAGuard device_guard(X.device());
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&
                          is_aligned_16(Y);
    if (can_vectorize) {
        rmsnorm_vectorized_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    } else {
        rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),
            rows, cols, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
