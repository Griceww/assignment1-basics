from einops import einsum
import torch
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        g = torch.empty(
            (d_model),
            device=device,
            dtype=dtype
        )
        torch.nn.init.constant_(g, 1)
        self.g = torch.nn.Parameter(g)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        x (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        result = einsum(x, self.g, "batch ... d, d -> batch ... d")
        return result.to(in_dtype)