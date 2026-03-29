import torch
import math
from einops import einsum
class Linear(torch.nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None) :
    super().__init__()
    W = torch.empty(
      (out_features, in_features),
      device=device,
      dtype=dtype,
    )
    std = math.sqrt(2 / (in_features + out_features))
    torch.nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3*std, b=3*std)
    self.W = torch.nn.Parameter(W)
    
  def forward(self, x:torch.Tensor) ->torch.Tensor:
    return einsum(x, self.W, "batch ... in, out in -> batch ... out")