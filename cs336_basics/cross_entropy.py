import torch
from einops import rearrange
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Multi-class cross-entropy with mean over all batch-like positions.

    inputs: (..., num_classes) unnormalized logits
    targets: (...) int64 class indices, same shape as inputs[..., 0]
    """
    last = -1
    c_max = inputs.amax(dim=last, keepdim=True)
    shifted = inputs - c_max
    log_partition = torch.log(torch.sum(torch.exp(shifted), dim=last))
    idx = rearrange(targets, "... -> ... ()")
    target_logits = rearrange(torch.take_along_dim(inputs, idx, dim=last), "... () -> ...")
    per_position = -target_logits + rearrange(c_max, "... () -> ...") + log_partition
    return per_position.mean()
