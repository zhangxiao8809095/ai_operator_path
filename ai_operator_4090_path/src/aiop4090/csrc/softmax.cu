#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

namespace {

__forceinline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__forceinline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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

__forceinline__ __device__ float block_reduce_max(float val, float* smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < warp_count) ? smem[lane] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

struct OnlineSoftmaxState {
    float max_val;
    float sum_val;
};

__forceinline__ __device__ OnlineSoftmaxState combine_online(OnlineSoftmaxState a,
                                                            OnlineSoftmaxState b) {
    if (a.sum_val == 0.0f) return b;
    if (b.sum_val == 0.0f) return a;
    float max_val = fmaxf(a.max_val, b.max_val);
    float sum_val = a.sum_val * expf(a.max_val - max_val) +
                    b.sum_val * expf(b.max_val - max_val);
    return {max_val, sum_val};
}

__forceinline__ __device__ OnlineSoftmaxState warp_reduce_online(OnlineSoftmaxState state) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        OnlineSoftmaxState other{
            __shfl_down_sync(0xffffffff, state.max_val, offset),
            __shfl_down_sync(0xffffffff, state.sum_val, offset),
        };
        state = combine_online(state, other);
    }
    return state;
}

__forceinline__ __device__ OnlineSoftmaxState block_reduce_online(OnlineSoftmaxState state,
                                                                 float* smem_max,
                                                                 float* smem_sum) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    state = warp_reduce_online(state);
    if (lane == 0) {
        smem_max[warp_id] = state.max_val;
        smem_sum[warp_id] = state.sum_val;
    }
    __syncthreads();

    state.max_val = (threadIdx.x < warp_count) ? smem_max[lane] : -FLT_MAX;
    state.sum_val = (threadIdx.x < warp_count) ? smem_sum[lane] : 0.0f;
    if (warp_id == 0) state = warp_reduce_online(state);
    if (threadIdx.x == 0) {
        smem_max[0] = state.max_val;
        smem_sum[0] = state.sum_val;
    }
    __syncthreads();
    return {smem_max[0], smem_sum[0]};
}

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

__global__ void softmax_warp_reduce_kernel(const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, X[row * cols + col]);
    }
    float max_val = block_reduce_max(local_max, smem);

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(X[row * cols + col] - max_val);
    }
    float sum_val = block_reduce_sum(local_sum, smem);

    for (int col = tid; col < cols; col += blockDim.x) {
        Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;
    }
}

__global__ void softmax_online_kernel(const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      int rows, int cols) {
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    OnlineSoftmaxState state{-FLT_MAX, 0.0f};
    for (int col = tid; col < cols; col += blockDim.x) {
        float x = X[row * cols + col];
        OnlineSoftmaxState item{x, 1.0f};
        state = combine_online(state, item);
    }
    state = block_reduce_online(state, smem_max, smem_sum);

    for (int col = tid; col < cols; col += blockDim.x) {
        Y[row * cols + col] = expf(X[row * cols + col] - state.max_val) / state.sum_val;
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

torch::Tensor softmax_warp_reduce(torch::Tensor X) {
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    softmax_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
    return Y;
}

torch::Tensor softmax_online(torch::Tensor X) {
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    int rows = static_cast<int>(X.size(0));
    int cols = static_cast<int>(X.size(1));
    auto Y = torch::empty_like(X);
    if (rows == 0 || cols == 0) return Y;
    int block = 256;
    softmax_online_kernel<<<rows, block, 64 * sizeof(float)>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
    return Y;
}
