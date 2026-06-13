#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

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
