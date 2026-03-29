import torch
from einops import einsum, rearrange
from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
import math

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    mask: torch.Tensor = None) :
    """
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"]

    return ( ... )
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... query dk, ... keys dk -> ... query keys") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, "... query keys, ... keys dv -> ... query dv")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 rope: RotaryPositionalEmbedding | None = None,
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        k = rearrange(k, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        v = rearrange(v, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        causal_mask = ~causal_mask

        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        attn_out = rearrange(attn_out, "... head seq d_v -> ... seq (head d_v)")
        return self.out_proj(attn_out)
