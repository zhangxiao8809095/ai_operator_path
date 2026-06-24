#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

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

__global__ void gemm_tiled_padding_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

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

constexpr int RT4_TILE = 16;
constexpr int RT4_RM = 4;
constexpr int RT4_RN = 4;

__global__ void gemm_regtile4x4_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ float As[RT4_TILE * RT4_RM][RT4_TILE];
    __shared__ float Bs[RT4_TILE][RT4_TILE * RT4_RN];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int base_row = blockIdx.y * (RT4_TILE * RT4_RM) + local_row * RT4_RM;
    int base_col = blockIdx.x * (RT4_TILE * RT4_RN) + local_col * RT4_RN;

    float acc[RT4_RM][RT4_RN] = {};

    for (int t = 0; t < K; t += RT4_TILE) {
        #pragma unroll
        for (int r = 0; r < RT4_RM; ++r) {
            int row = base_row + r;
            int col = t + local_col;
            As[local_row * RT4_RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }

        #pragma unroll
        for (int c = 0; c < RT4_RN; ++c) {
            int row = t + local_row;
            int col = base_col + c;
            Bs[local_row][local_col * RT4_RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RT4_TILE; ++kk) {
            float a_vals[RT4_RM];
            float b_vals[RT4_RN];

            #pragma unroll
            for (int r = 0; r < RT4_RM; ++r) {
                a_vals[r] = As[local_row * RT4_RM + r][kk];
            }

            #pragma unroll
            for (int c = 0; c < RT4_RN; ++c) {
                b_vals[c] = Bs[kk][local_col * RT4_RN + c];
            }

            #pragma unroll
            for (int r = 0; r < RT4_RM; ++r) {
                #pragma unroll
                for (int c = 0; c < RT4_RN; ++c) {
                    acc[r][c] += a_vals[r] * b_vals[c];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < RT4_RM; ++r) {
        #pragma unroll
        for (int c = 0; c < RT4_RN; ++c) {
            int row = base_row + r;
            int col = base_col + c;
            if (row < M && col < N) C[row * N + col] = acc[r][c];
        }
    }
}

__global__ void gemm_vectorized_float4_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int base_col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row >= M) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    bool can_vectorize_b = (N % 4 == 0) && (base_col + 3 < N);

    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        if (can_vectorize_b) {
            float4 b = reinterpret_cast<const float4*>(B + k * N + base_col)[0];
            acc0 += a * b.x;
            acc1 += a * b.y;
            acc2 += a * b.z;
            acc3 += a * b.w;
        } else {
            if (base_col + 0 < N) acc0 += a * B[k * N + base_col + 0];
            if (base_col + 1 < N) acc1 += a * B[k * N + base_col + 1];
            if (base_col + 2 < N) acc2 += a * B[k * N + base_col + 2];
            if (base_col + 3 < N) acc3 += a * B[k * N + base_col + 3];
        }
    }

    if (base_col + 0 < N) C[row * N + base_col + 0] = acc0;
    if (base_col + 1 < N) C[row * N + base_col + 1] = acc1;
    if (base_col + 2 < N) C[row * N + base_col + 2] = acc2;
    if (base_col + 3 < N) C[row * N + base_col + 3] = acc3;
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_WARPS_PER_BLOCK = 4;

__global__ void gemm_wmma_fp16_kernel(const half* __restrict__ A,
                                      const half* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    int warp_id = threadIdx.y;
    int tile_m = blockIdx.y * WMMA_WARPS_PER_BLOCK + warp_id;
    int tile_n = blockIdx.x;
    int row = tile_m * WMMA_M;
    int col = tile_n * WMMA_N;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * K + k0, K);
        wmma::load_matrix_sync(b_frag, B + k0 * N + col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + row * N + col, acc_frag, N, wmma::mem_row_major);
}

void check_gemm_shape_and_device(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");
}

void check_gemm_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    check_gemm_shape_and_device(A, B);
    CHECK_FLOAT32(A);
    CHECK_FLOAT32(B);
}

void check_gemm_fp16_compatible_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    check_gemm_shape_and_device(A, B);
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same dtype");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 || A.scalar_type() == torch::kFloat16,
                "A and B must be float32 or float16");
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
    gemm_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
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
    gemm_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_tiled_padding(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div_int(N, TILE), ceil_div_int(M, TILE));
    gemm_tiled_padding_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
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
    gemm_regtile2x2_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_regtile4x4(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(RT4_TILE, RT4_TILE);
    dim3 grid(ceil_div_int(N, RT4_TILE * RT4_RN), ceil_div_int(M, RT4_TILE * RT4_RM));
    gemm_regtile4x4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_vectorized_float4(torch::Tensor A, torch::Tensor B) {
    check_gemm_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(ceil_div_int(N, block.x * 4), ceil_div_int(M, block.y));
    gemm_vectorized_float4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_wmma_fp16(torch::Tensor A, torch::Tensor B) {
    check_gemm_fp16_compatible_inputs(A, B);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        torch::Tensor A_float = A.scalar_type() == torch::kFloat32 ? A : A.to(torch::kFloat32);
        torch::Tensor B_float = B.scalar_type() == torch::kFloat32 ? B : B.to(torch::kFloat32);
        dim3 block(TILE, TILE);
        dim3 grid(ceil_div_int(N, TILE), ceil_div_int(M, TILE));
        gemm_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            A_float.data_ptr<float>(), B_float.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
        return C;
    }

    torch::Tensor A_half = A.scalar_type() == torch::kFloat16 ? A : A.to(torch::kFloat16);
    torch::Tensor B_half = B.scalar_type() == torch::kFloat16 ? B : B.to(torch::kFloat16);
    dim3 block(32, WMMA_WARPS_PER_BLOCK);
    dim3 grid(ceil_div_int(N, WMMA_N), ceil_div_int(ceil_div_int(M, WMMA_M), WMMA_WARPS_PER_BLOCK));
    gemm_wmma_fp16_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(A_half.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_half.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        M, N, K);
    return C;
}
