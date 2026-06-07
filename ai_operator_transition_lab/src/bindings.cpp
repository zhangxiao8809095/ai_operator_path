#include <torch/extension.h>

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor softmax_row(torch::Tensor X);
torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "Naive FP32 GEMM CUDA");
    m.def("gemm_tiled", &gemm_tiled, "Shared-memory tiled FP32 GEMM CUDA");
    m.def("softmax_row", &softmax_row, "Row-wise FP32 softmax CUDA");
    m.def("layernorm_row", &layernorm_row, "Row-wise FP32 LayerNorm CUDA");
    m.def("rmsnorm_row", &rmsnorm_row, "Row-wise FP32 RMSNorm CUDA");
    m.def("attention_naive", &attention_naive, "Naive single-head FP32 attention CUDA");
}
