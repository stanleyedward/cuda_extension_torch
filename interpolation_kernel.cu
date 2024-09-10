#include <torch/extension.h>

torch::Tensor trilinear_forward_cu(
    torch::Tensor features,
    torch::Tensor points
){
    return features;
}