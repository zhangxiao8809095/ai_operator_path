#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.numel() > 0, "input must be non-empty");
}

__global__ void identity_kernel(const float* input, float* output, int64_t count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index];
}

__global__ void out_of_bounds_kernel(const float* input, float* output, int64_t count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index <= count) {
        output[index] = index < count ? input[index] : 0.0f;  // index == count is intentional.
    }
}

__global__ void shared_race_kernel(const float* input, float* output, int64_t count) {
    __shared__ float shared_value;
    int index = threadIdx.x % count;
    shared_value = input[index];  // Intentional WAW race: every thread writes the same location.
    if (threadIdx.x == 0) output[0] = shared_value;  // Intentional RAW race without a barrier.
}

__global__ void uninitialized_read_kernel(const float* scratch,
                                          const float* input,
                                          float* output,
                                          int64_t count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] + scratch[index];
}

__global__ void illegal_address_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally dereference an invalid address. This is separate from
        // the one-element OOB used by memcheck: it must raise an asynchronous
        // execution error for the CUDA_LAUNCH_BLOCKING comparison lab.
        volatile int* invalid = reinterpret_cast<volatile int*>(
            static_cast<std::uintptr_t>(1));
        *invalid = 2026;
    }
}

int blocks_for(int64_t count, int threads) {
    return static_cast<int>((count + threads - 1) / threads);
}

}  // namespace

torch::Tensor fault_identity(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    identity_kernel<<<blocks_for(input.numel(), threads), threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor fault_out_of_bounds(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    out_of_bounds_kernel<<<blocks_for(input.numel() + 1, threads), threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor fault_shared_race(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::zeros_like(input);
    shared_race_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor fault_uninitialized_read(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    float* scratch = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&scratch, input.numel() * sizeof(float)));
    constexpr int threads = 256;
    uninitialized_read_kernel<<<blocks_for(input.numel(), threads), threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        scratch, input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaFree(scratch));
    return output;
}

void fault_invalid_launch(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    identity_kernel<<<1, 2048, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fault_illegal_address(torch::Tensor input) {
    check_input(input);
    c10::cuda::CUDAGuard guard(input.device());
    illegal_address_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
