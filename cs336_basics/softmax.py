from torch import Tensor
import torch

def softmax(in_features: Tensor, dim: int) -> Tensor:
    x_max = in_features.amax(dim=dim, keepdim=True)
    shifted = in_features - x_max
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)