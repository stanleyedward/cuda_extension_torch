#include <torch/extension.h>

template <typename scalar_t> // scalar_t is a placeholder dtype so we dont have to explicitly define the dtype
__global__ void trilinear_forward_kernel(
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> features,
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output
){

}

torch::Tensor trilinear_forward_cu(
    torch::Tensor features,
    torch::Tensor points        
){  
    const int N = features.size(0), F = features.size(2); //  num of cubes and dimension of features in each vertex

    // feat_interp_output = torch.zeros(N, F, dtype=torch.float32, device='cuda:0')
    // torch::zeros({N,F}, torch::dtype(torch::kInt32).device(features.device())); // change tensors dtype and device
    torch::Tensor featInterpOutput = torch::empty({N, F}, features.options()); // options sets dtype and device same as features
    const dim3 numThreadsPerBlock(16, 16, 1); //256 threads in each dim
    const dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (F + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    // instantiate kernel
    AT_DISPATCH_FLOATING_TYPES(features.type(), "trilinear_forward_cu()", 
    ([&] {
        trilinear_forward_kernel<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
            // packed accessor is type conversion for tensors so cuda can manipulate them (not needed by primitive cpp dtypes)
            // restrictPtrTraits: to prevent memory overlay of tensors
            // size_t:  how many steps to take btw each element 
            features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),         
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            featInterpOutput.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    })  
    );

    return featInterpOutput;
}