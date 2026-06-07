#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// ---------------- GEMM ----------------

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

template<int TILE>
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1]; // +1 avoids many shared-memory bank conflicts

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
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(div_up(N, block.x), div_up(M, block.y));
    gemm_naive_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(div_up(N, TILE), div_up(M, TILE));
    gemm_tiled_kernel<TILE><<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

// ---------------- Row Softmax ----------------

__global__ void softmax_row_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int rows, int cols) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, X[row * cols + c]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float m = smax[0];

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += expf(X[row * cols + c] - m);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    for (int c = tid; c < cols; c += blockDim.x) {
        Y[row * cols + c] = expf(X[row * cols + c] - m) * inv_sum;
    }
}

torch::Tensor softmax_row(torch::Tensor X) {
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    int rows = X.size(0), cols = X.size(1);
    auto Y = torch::empty_like(X);
    int threads = 256;
    size_t smem = threads * 2 * sizeof(float);
    softmax_row_kernel<<<rows, threads, smem>>>(X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
    return Y;
}

// ---------------- LayerNorm / RMSNorm ----------------

__global__ void layernorm_row_kernel(const float* __restrict__ X,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ Y,
                                     int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) local_sum += X[row * cols + c];
    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float mean = smem[0] / cols;

    float local_var = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = X[row * cols + c] - mean;
        local_var += v * v;
    }
    smem[tid] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / cols + eps);

    for (int c = tid; c < cols; c += blockDim.x) {
        float norm = (X[row * cols + c] - mean) * inv_std;
        Y[row * cols + c] = norm * gamma[c] + beta[c];
    }
}

__global__ void rmsnorm_row_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ Y,
                                   int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float local_ss = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = X[row * cols + c];
        local_ss += v * v;
    }
    smem[tid] = local_ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / cols + eps);
    for (int c = tid; c < cols; c += blockDim.x) {
        Y[row * cols + c] = X[row * cols + c] * inv_rms * gamma[c];
    }
}

torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(X); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma and beta must be 1D");
    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), "gamma/beta size mismatch");
    int rows = X.size(0), cols = X.size(1);
    auto Y = torch::empty_like(X);
    int threads = 256;
    layernorm_row_kernel<<<rows, threads, threads * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    return Y;
}

torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps) {
    CHECK_INPUT(X); CHECK_INPUT(gamma);
    TORCH_CHECK(X.dim() == 2, "X must be 2D [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1), "gamma size mismatch");
    int rows = X.size(0), cols = X.size(1);
    auto Y = torch::empty_like(X);
    int threads = 256;
    rmsnorm_row_kernel<<<rows, threads, threads * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    return Y;
}

// ---------------- Naive single-head attention ----------------
// Q, K, V: [S, D], output O: [S, D]. Debug kernel, not production-grade.

__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int S, int D,
                                       bool causal) {
    extern __shared__ float scores[];
    int q = blockIdx.x;
    int tid = threadIdx.x;
    if (q >= S) return;

    float scale = rsqrtf((float)D);

    // Step 1: scores[q, k] = dot(Q[q], K[k]) / sqrt(D)
    for (int k = tid; k < S; k += blockDim.x) {
        if (causal && k > q) {
            scores[k] = -INFINITY;
        } else {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += Q[q * D + d] * K[k * D + d];
            }
            scores[k] = dot * scale;
        }
    }
    __syncthreads();

    // Step 2: softmax max
    __shared__ float reduce_buf[256];
    float local_max = -INFINITY;
    for (int k = tid; k < S; k += blockDim.x) local_max = fmaxf(local_max, scores[k]);
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        __syncthreads();
    }
    float m = reduce_buf[0];

    // Step 3: softmax sum
    float local_sum = 0.0f;
    for (int k = tid; k < S; k += blockDim.x) local_sum += expf(scores[k] - m);
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_buf[0];

    // Step 4: O[q, d] = sum_k softmax(scores[k]) * V[k, d]
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < S; ++k) {
            float p = expf(scores[k] - m) * inv_sum;
            acc += p * V[k * D + d];
        }
        O[q * D + d] = acc;
    }
}

torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2 && V.dim() == 2, "Q/K/V must be 2D [S, D]");
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "sequence length mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "hidden dim mismatch");
    int S = Q.size(0), D = Q.size(1);
    TORCH_CHECK(S <= 4096, "debug kernel supports S <= 4096 because scores use shared memory");
    auto O = torch::empty_like(Q);
    int threads = 256;
    size_t smem = S * sizeof(float);
    attention_naive_kernel<<<S, threads, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), S, D, causal);
    return O;
}
