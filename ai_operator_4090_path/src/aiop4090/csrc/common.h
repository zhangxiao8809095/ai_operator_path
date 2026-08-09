#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <climits>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)
#define CHECK_SAME_DEVICE(x, y) \
    TORCH_CHECK((x).get_device() == (y).get_device(), #x " and " #y " must be on the same CUDA device")
#define CHECK_DIM_FITS_INT(x, dim) \
    TORCH_CHECK((x).size(dim) <= INT_MAX, #x " dimension " #dim " exceeds int32 kernel indexing")

inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }
