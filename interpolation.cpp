/** */

#include <torch/extension.h>

// features: tensor of 8 vertices of the cube
// points: point inside the cube to interpolate to
torch::Tensor trilinear_interpolation(
    torch::Tensor features,
    torch::Tensor point)
{
    return features;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation); 
    // m.def("python_function_name", cpp_function_name)
}