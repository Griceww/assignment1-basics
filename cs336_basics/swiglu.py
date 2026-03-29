import torch

from cs336_basics.linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        # W1: (d_model -> d_ff), W3: (d_model -> d_ff), W2: (d_ff -> d_model)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # SwiGLU: W2( silu(W1x) * (W3x) )
        x1 = self.w1(in_features)
        x3 = self.w3(in_features)
        # SiLU(x) = x * sigmoid(x)
        return self.w2((x1 * torch.sigmoid(x1)) * x3)