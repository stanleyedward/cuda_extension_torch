/* Bridge between pytorch and CUDA */
#include <torch/extension.h>
#include "utils.h"

// features: tensor of 8 vertices of the cube
// points: point inside the cube to interpolate to
torch::Tensor trilinear_interpolation_forward(
    torch::Tensor features,
    torch::Tensor points)
{   
    CHECK_INPUT(features);
    CHECK_INPUT(points);
    return trilinear_forward_cu(features, points);
}

torch::Tensor trilinear_interpolation_backward(
    torch::Tensor dL_dfeatInterpOutput,
    torch::Tensor features,
    torch::Tensor points)
{
    CHECK_INPUT(dL_dfeatInterpOutput);
    CHECK_INPUT(features);
    CHECK_INPUT(points);
    return trilinear_backward_cu(dL_dfeatInterpOutput, features, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation_forward", &trilinear_interpolation_forward); 
    m.def("trilinear_interpolation_backward", &trilinear_interpolation_backward);
    // m.def("python_function_name", cpp_function_name)
}