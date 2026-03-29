import torch
class Embedding(torch.nn.Module) :
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        )
        std = 1.0
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]