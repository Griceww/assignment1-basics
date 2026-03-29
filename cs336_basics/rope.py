import torch
from einops import einsum


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires even d_k.")
        # One frequency per dimension pair: theta^(-2i/d_k) for i = 0, 2, ..., d_k-2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = einsum(positions, inv_freq, "seq, dim -> seq dim")
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x               '(..., seq_len, d_k)'
        token positions '(..., seq_len)'
        return          '(..., seq_len, d_k)'
        """
        cos = self.cos_cached[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype, device=x.device)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., ::2] = y1
        out[..., 1::2] = y2
        return out
