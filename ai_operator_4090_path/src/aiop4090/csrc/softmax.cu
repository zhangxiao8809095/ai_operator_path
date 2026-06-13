#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

namespace {

__global__ void softmax_row_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int rows, int cols) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, X[row * cols + col]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(X[row * cols + col] - max_val);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float sum_val = ssum[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;
    }
}

} // namespace

torch::Tensor softmax_row(torch::Tensor X) {
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    int block = 256;
    softmax_row_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
    return Y;
}
