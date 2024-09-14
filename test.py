import time
import torch #needed to import interpolation
import interpolation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 65536
F = 256
randn = torch.randn(N, 8, F, device=device)
features = randn.clone().requires_grad_()
features2 = randn.clone().requires_grad_()
points = torch.randn(N, 3, device=device) * 2 - 1  

class TrilinearInterpolationCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, points):
        featInterp = interpolation.trilinear_interpolation_forward(features, points)
        ctx.save_for_backward(features, points)
        return featInterp
    
    @staticmethod
    def backward(ctx, dL_dfeatInterpOutput):
        features, points = ctx.saved_tensors
        dL_dFeatures = interpolation.trilinear_interpolation_backward(dL_dfeatInterpOutput.contiguous(), features, points)
        return dL_dFeatures, None # second is None as 2nd input (points) is constant and has no gradients

def trilinear_interpolation_py(feats, points):
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] + 
                          b*feats[:, 1] + 
                          c*feats[:, 2] + 
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available, you need a GPU for this"
    try:
        t = time.time()
        # output_cuda = interpolation.trilinear_interpolation_foward(features2, points) 
        output_cuda = TrilinearInterpolationCuda.apply(features2, points) #since we have made the autograd class
        print(f"cuda fw time: {time.time() - t}")
        torch.cuda.synchronize()
        print(f"Is cuda trainable?: {output_cuda.requires_grad}")
        print(f"cuda shape; {output_cuda.shape}")
        
        t2 = time.time()
        output_py = trilinear_interpolation_py(features, points)
        print(f"pytorch fw time: {time.time() - t2}")
        torch.cuda.synchronize()
        print(f"Is pytorch trainable?: {output_py.requires_grad}")
        print("forward tensors close?:", torch.allclose(output_cuda, output_py, rtol=1e-6, atol=1e-5))
        
        t = time.time()
        loss = output_py.sum()
        loss.backward()
        print(f"pytorch bw time: {time.time() - t}")
        torch.cuda.synchronize()
        
        t2 = time.time()
        loss_cuda = output_cuda.sum()
        loss_cuda.backward()  
        print(f"cuda bw time: {time.time() - t2}")
        torch.cuda.synchronize()
        
        print(f"backward tensors close?: {torch.allclose(features.grad, features2.grad, rtol=1e-6, atol=1e-5)}")
    except AssertionError as ae:
        print(f"Error occured: {ae}")