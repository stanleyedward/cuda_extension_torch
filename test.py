import torch #needed to import interpolation
import interpolation

features = torch.ones(2)
point = torch.randn(2)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert torch.cuda.is_available(), "CUDA is not available, you need a GPU for this"
    
    try:
        output = interpolation.trilinear_interpolation(features.to(device=device), point.to(device=device))
        print(output)
    except AssertionError as ae:
        print(f"Error occured: {ae}")