#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// 每个线程计算 C 中相邻 2x2 输出。这个版本用于展示 register tiling 的思想，
// 不是最终极致版本。后续可继续扩展为 4x4/8x4 并引入更细的 shared memory swizzle。
constexpr int RT_TILE = 16;
constexpr int RM = 2;
constexpr int RN = 2;

__global__ void gemm_regtile2x2_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ float As[RT_TILE * RM][RT_TILE];
    __shared__ float Bs[RT_TILE][RT_TILE * RN];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int base_row = blockIdx.y * (RT_TILE * RM) + local_row * RM;
    int base_col = blockIdx.x * (RT_TILE * RN) + local_col * RN;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    for (int t = 0; t < K; t += RT_TILE) {
        // load 2 rows of A per thread
        for (int r = 0; r < RM; ++r) {
            int row = base_row + r;
            int col = t + local_col;
            As[local_row * RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        // load 2 cols of B per thread
        for (int c = 0; c < RN; ++c) {
            int row = t + local_row;
            int col = base_col + c;
            Bs[local_row][local_col * RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RT_TILE; ++kk) {
            float a0 = As[local_row * RM + 0][kk];
            float a1 = As[local_row * RM + 1][kk];
            float b0 = Bs[kk][local_col * RN + 0];
            float b1 = Bs[kk][local_col * RN + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    if (base_row + 0 < M && base_col + 0 < N) C[(base_row + 0) * N + base_col + 0] = acc00;
    if (base_row + 0 < M && base_col + 1 < N) C[(base_row + 0) * N + base_col + 1] = acc01;
    if (base_row + 1 < M && base_col + 0 < N) C[(base_row + 1) * N + base_col + 0] = acc10;
    if (base_row + 1 < M && base_col + 1 < N) C[(base_row + 1) * N + base_col + 1] = acc11;
}

void check_gemm_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");
}

} // namespace

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(ceil_div_int(N, block.x), ceil_div_int(M, block.y));
    gemm_naive_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div_int(N, TILE), ceil_div_int(M, TILE));
    gemm_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_regtile2x2(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(RT_TILE, RT_TILE);
    dim3 grid(ceil_div_int(N, RT_TILE * RN), ceil_div_int(M, RT_TILE * RM));
    gemm_regtile2x2_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
