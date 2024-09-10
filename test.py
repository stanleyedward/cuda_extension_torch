import torch #needed to import interpolation
import interpolation

features = torch.ones(2)
point = torch.randn(2)

if __name__ == "__main__":
    
    output = interpolation.trilinear_interpolation(features, point)
    print(output)