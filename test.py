import time
import torch #needed to import interpolation
import interpolation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 65536
F = 256

features = torch.randn(N, 8, F, device=device).requires_grad_()
points = torch.randn(N, 3, device=device) * 2 - 1  

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
        output_cuda = interpolation.trilinear_interpolation(features, points)
        print(f"cuda time: {time.time() - t}")
        torch.cuda.synchronize()
        print(f"Is cuda trainable?: {output_cuda.requires_grad}")
        print(f"cuda shape; {output_cuda.shape}")
        
        t2 = time.time()
        output_py = trilinear_interpolation_py(features, points)
        print(f"pytorch time: {time.time() - t2}")
        torch.cuda.synchronize()
        print(f"Is pytorch trainable?: {output_py.requires_grad}")
        print(torch.allclose(output_cuda, output_py, rtol=1e-6, atol=1e-5))
    except AssertionError as ae:
        print(f"Error occured: {ae}")