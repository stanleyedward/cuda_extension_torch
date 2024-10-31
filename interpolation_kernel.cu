// #include <torch/extension.h>

// template <typename scalar_t> // scalar_t is a placeholder dtype so we dont have to explicitly define the dtype
// __global__ void trilinear_forward_kernel(
//             const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> features,
//             const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
//             torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output
// ){
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     const int f = blockDim.y * blockIdx.y + threadIdx.y;

//     if (n < features.size(0) && f < features.size(2)){
//         //since the range for points is [-1, 1] we div by 2 to normalize
//         const scalar_t u = (points[n][0] + 1)/2;
//         const scalar_t v = (points[n][1] + 1)/2;
//         const scalar_t w = (points[n][2] + 1)/2;    

//         //interpolation coef
//         const scalar_t a = (1-v)*(1-w);
//         const scalar_t b = (1-v)*w;
//         const scalar_t c = v*(1-w);
//         const scalar_t d = 1-a-b-c;

//         output[n][f] = (1-u)*(a*features[n][0][f] +
//                         b*features[n][1][f] +
//                         c*features[n][2][f] + 
//                         d*features[n][3][f]) +
//                         u*(a*features[n][4][f]+
//                         b*features[n][5][f]+
//                         c*features[n][6][f]+
//                         d*features[n][7][f]);
//     }
// }


// torch::Tensor trilinear_forward_cu(
//     const torch::Tensor features,
//     const torch::Tensor points        
// ){  
//     const int N = features.size(0), F = features.size(2); //  num of cubes and dimension of features in each vertex

//     // feat_interp_output = torch.zeros(N, F, dtype=torch.float32, device='cuda:0')
//     // torch::zeros({N,F}, torch::dtype(torch::kInt32).device(features.device())); // change tensors dtype and device
//     torch::Tensor featInterpOutput = torch::empty({N, F}, features.options()); // options sets dtype and device same as features
//     const dim3 numThreadsPerBlock(16, 16, 1); //256 threads in each dim
//     const dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (F + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

//     // instantiate kernel
//     AT_DISPATCH_FLOATING_TYPES(features.type(), "trilinear_forward_cu()", 
//     ([&] {
//         trilinear_forward_kernel<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
//             // packed accessor is type conversion for tensors so cuda can manipulate them (not needed by primitive cpp dtypes)
//             // restrictPtrTraits: to prevent memory overlay of tensors
//             // size_t:  how many steps to take btw each element 
//             features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),         
//             points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
//             featInterpOutput.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
//         );
//     })  
//     );

//     return featInterpOutput;
// }

// template <typename scalar_t>
// __global__ void trilinear_backward_kernel(
//             const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> features,         
//             const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
//             const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeatInterpOutput,
//             torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dFeatures
// ){

//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     const int f = blockDim.y * blockIdx.y + threadIdx.y;

//     if (n < features.size(0) && f < features.size(2)){
//         //since the range for points is [-1, 1] we div by 2 to normalize
//         const scalar_t u = (points[n][0] + 1)/2;
//         const scalar_t v = (points[n][1] + 1)/2;
//         const scalar_t w = (points[n][2] + 1)/2;    

//         //interpolation coef
//         const scalar_t a = (1-v)*(1-w);
//         const scalar_t b = (1-v)*w;
//         const scalar_t c = v*(1-w);
//         const scalar_t d = 1-a-b-c;

//         dL_dFeatures[n][0][f] = (1-u) * a * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][1][f] = (1-u) * b * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][2][f] = (1-u) * c * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][3][f] = (1-u) * d * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][4][f] = u * a * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][5][f] = u * b * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][6][f] = u * c * dL_dfeatInterpOutput[n][f];
//         dL_dFeatures[n][7][f] = u * d * dL_dfeatInterpOutput[n][f];
//     }
// }


// torch::Tensor trilinear_backward_cu(
//     const torch::Tensor dL_dfeatInterpOutput,
//     const torch::Tensor features,
//     const torch::Tensor points        
// ){  
//     const int N = features.size(0), F = features.size(2); //  num of cubes and dimension of features in each vertex

//     torch::Tensor dL_dFeatures = torch::empty({N, 8, F}, features.options()); 
//     const dim3 numThreadsPerBlock(16, 16, 1); 
//     const dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (F + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

//     // instantiate kernel
//     AT_DISPATCH_FLOATING_TYPES(features.type(), "trilinear_backward_cu()", 
//     ([&] {
//         trilinear_backward_kernel<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
//             // packed accessor is type conversion for tensors so cuda can manipulate them (not needed by primitive cpp dtypes)
//             // restrictPtrTraits: to prevent memory overlay of tensors
//             // size_t:  how many steps to take btw each element 
//             features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),         
//             points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
//             dL_dfeatInterpOutput.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
//             dL_dFeatures.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
//         );
//     })  
//     );

//     return dL_dFeatures;
// }


#include <torch/extension.h>

__global__ void trilinear_forward_kernel(
    int N,              // number of points
    int F,              // feature dimension
    const float* features,
    const float* points,
    float* output
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && f < F) {
        // Get point coordinates (normalized from [-1,1] to [0,1])
        const float u = (points[n * 3 + 0] + 1) / 2;
        const float v = (points[n * 3 + 1] + 1) / 2;
        const float w = (points[n * 3 + 2] + 1) / 2;

        // Interpolation coefficients
        const float a = (1-v) * (1-w);
        const float b = (1-v) * w;
        const float c = v * (1-w);
        const float d = 1-a-b-c;

        // Calculate feature index offsets
        const int feat_idx_base = n * 8 * F;  // Base index for the current point's features
        
        // Calculate interpolated value
        output[n * F + f] = 
            (1-u) * (
                a * features[feat_idx_base + 0 * F + f] +
                b * features[feat_idx_base + 1 * F + f] +
                c * features[feat_idx_base + 2 * F + f] + 
                d * features[feat_idx_base + 3 * F + f]
            ) +
            u * (
                a * features[feat_idx_base + 4 * F + f] +
                b * features[feat_idx_base + 5 * F + f] +
                c * features[feat_idx_base + 6 * F + f] +
                d * features[feat_idx_base + 7 * F + f]
            );
    }
}

at::Tensor trilinear_forward_cu(
    const at::Tensor features,
    const at::Tensor points
) {
    TORCH_CHECK(features.dim() == 3, "features must be 3D");
    TORCH_CHECK(points.dim() == 2, "points must be 2D");
    TORCH_CHECK(points.size(1) == 3, "points must have 3 coordinates");
    TORCH_CHECK(features.dtype() == at::kFloat);
    TORCH_CHECK(points.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(features.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(points.device().type() == at::DeviceType::CUDA);

    const int N = features.size(0);  // number of points
    const int F = features.size(2);  // feature dimension

    // Ensure inputs are contiguous
    at::Tensor features_contig = features.contiguous();
    at::Tensor points_contig = points.contiguous();

    // Create output tensor
    at::Tensor output = torch::empty({N, F}, features.options());

    // Get raw pointers
    const float* features_ptr = features_contig.data_ptr<float>();
    const float* points_ptr = points_contig.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (F + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    trilinear_forward_kernel<<<numBlocks, threadsPerBlock>>>(
        N, F, features_ptr, points_ptr, output_ptr
    );

    return output;
}

__global__ void trilinear_backward_kernel(
    int N,              // number of points
    int F,              // feature dimension
    const float* features,
    const float* points,
    const float* grad_output,
    float* grad_features
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && f < F) {
        const float u = (points[n * 3 + 0] + 1) / 2;
        const float v = (points[n * 3 + 1] + 1) / 2;
        const float w = (points[n * 3 + 2] + 1) / 2;

        const float a = (1-v) * (1-w);
        const float b = (1-v) * w;
        const float c = v * (1-w);
        const float d = 1-a-b-c;

        const float grad = grad_output[n * F + f];
        const int feat_idx_base = n * 8 * F;

        grad_features[feat_idx_base + 0 * F + f] = (1-u) * a * grad;
        grad_features[feat_idx_base + 1 * F + f] = (1-u) * b * grad;
        grad_features[feat_idx_base + 2 * F + f] = (1-u) * c * grad;
        grad_features[feat_idx_base + 3 * F + f] = (1-u) * d * grad;
        grad_features[feat_idx_base + 4 * F + f] = u * a * grad;
        grad_features[feat_idx_base + 5 * F + f] = u * b * grad;
        grad_features[feat_idx_base + 6 * F + f] = u * c * grad;
        grad_features[feat_idx_base + 7 * F + f] = u * d * grad;
    }
}

at::Tensor trilinear_backward_cu(
    const at::Tensor grad_output,
    const at::Tensor features,
    const at::Tensor points
) {
    const int N = features.size(0);
    const int F = features.size(2);

    // Ensure inputs are contiguous
    at::Tensor grad_output_contig = grad_output.contiguous();
    at::Tensor features_contig = features.contiguous();
    at::Tensor points_contig = points.contiguous();

    // Create gradient tensor
    at::Tensor grad_features = torch::empty({N, 8, F}, features.options());

    // Get raw pointers
    const float* grad_output_ptr = grad_output_contig.data_ptr<float>();
    const float* features_ptr = features_contig.data_ptr<float>();
    const float* points_ptr = points_contig.data_ptr<float>();
    float* grad_features_ptr = grad_features.data_ptr<float>();

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (F + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    trilinear_backward_kernel<<<numBlocks, threadsPerBlock>>>(
        N, F, features_ptr, points_ptr, grad_output_ptr, grad_features_ptr
    );

    return grad_features;
}