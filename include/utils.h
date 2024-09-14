#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_forward_cu(
    torch::Tensor features,
    torch::Tensor points
);

torch::Tensor trilinear_backward_cu(
    torch::Tensor dl_dInterpOutput,
    torch::Tensor features,
    torch::Tensor points
);