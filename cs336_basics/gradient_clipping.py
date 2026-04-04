from collections.abc import Iterable

import torch


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm_sq = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for grad in grads:
        total_norm_sq = total_norm_sq + grad.pow(2).sum()
    total_norm = total_norm_sq.sqrt()

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)
