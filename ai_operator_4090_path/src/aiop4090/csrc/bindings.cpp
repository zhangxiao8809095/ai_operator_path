#include <torch/extension.h>

torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_tiled_padding(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_regtile2x2(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_regtile4x4(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_vectorized_float4(torch::Tensor A, torch::Tensor B);
torch::Tensor gemm_wmma_fp16(torch::Tensor A, torch::Tensor B);
torch::Tensor softmax_row(torch::Tensor X);
torch::Tensor softmax_block_reduce(torch::Tensor X);
torch::Tensor softmax_warp_reduce(torch::Tensor X);
torch::Tensor softmax_online(torch::Tensor X);
torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor layernorm_block_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor layernorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor layernorm_vectorized(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor rmsnorm_block_reduce(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor rmsnorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor rmsnorm_vectorized(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor rmsnorm_vectorized_float4(torch::Tensor X, torch::Tensor gamma, double eps);
torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);
torch::Tensor attention_causal_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor attention_kv_cache_decode(torch::Tensor Q, torch::Tensor K_cache, torch::Tensor V_cache, int kv_len);
torch::Tensor attention_tiled_online_softmax(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "Naive FP32 GEMM CUDA");
    m.def("gemm_tiled", &gemm_tiled, "Shared-memory tiled FP32 GEMM CUDA");
    m.def("gemm_tiled_padding", &gemm_tiled_padding, "Padded shared-memory tiled FP32 GEMM CUDA");
    m.def("gemm_regtile2x2", &gemm_regtile2x2, "2x2 register-tiled FP32 GEMM CUDA");
    m.def("gemm_regtile4x4", &gemm_regtile4x4, "4x4 register-tiled FP32 GEMM CUDA");
    m.def("gemm_vectorized_float4", &gemm_vectorized_float4, "Float4 vectorized FP32 GEMM CUDA");
    m.def("gemm_wmma_fp16", &gemm_wmma_fp16, "WMMA FP16 GEMM CUDA with FP32 output");
    m.def("softmax_row", &softmax_row, "Row-wise softmax CUDA");
    m.def("softmax_block_reduce", &softmax_block_reduce, "Row-wise softmax CUDA with shared-memory block reduction");
    m.def("softmax_warp_reduce", &softmax_warp_reduce, "Row-wise softmax CUDA with warp-shuffle reduction");
    m.def("softmax_online", &softmax_online, "Row-wise online softmax CUDA");
    m.def("layernorm_row", &layernorm_row, "Row-wise LayerNorm CUDA");
    m.def("layernorm_block_reduce", &layernorm_block_reduce, "Row-wise LayerNorm CUDA with block reduction");
    m.def("layernorm_warp_reduce", &layernorm_warp_reduce, "Row-wise LayerNorm CUDA with warp-shuffle reduction");
    m.def("layernorm_vectorized", &layernorm_vectorized, "Float4 vectorized row-wise LayerNorm CUDA");
    m.def("rmsnorm_row", &rmsnorm_row, "Row-wise RMSNorm CUDA");
    m.def("rmsnorm_block_reduce", &rmsnorm_block_reduce, "Row-wise RMSNorm CUDA with block reduction");
    m.def("rmsnorm_warp_reduce", &rmsnorm_warp_reduce, "Row-wise RMSNorm CUDA with warp-shuffle reduction");
    m.def("rmsnorm_vectorized", &rmsnorm_vectorized, "Float4 vectorized row-wise RMSNorm CUDA");
    m.def("rmsnorm_vectorized_float4", &rmsnorm_vectorized_float4, "Float4 vectorized row-wise RMSNorm CUDA");
    m.def("attention_naive", &attention_naive, "Naive causal/non-causal attention CUDA");
    m.def("attention_causal_naive", &attention_causal_naive, "Naive causal attention CUDA");
    m.def("attention_kv_cache_decode", &attention_kv_cache_decode, "KV-cache decode attention CUDA");
    m.def("attention_tiled_online_softmax", &attention_tiled_online_softmax, "Tiled online-softmax attention CUDA");
}
