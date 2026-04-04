from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple, Union

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with assignment-specific update order."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step_tmp(self, closure=None):
        """Readable version using + - * / expressions."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scalar_dtype = (
            torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
        )

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=scalar_dtype)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                step_buf = state["step"]
                step_buf += 1
                t = int(step_buf.item())

                m.copy_(beta1 * m + (1 - beta1) * grad)
                v.copy_(beta2 * v + (1 - beta2) * (grad * grad))

                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                alpha_t = lr * (bc2**0.5) / bc1

                theta_old = p.clone()
                p.copy_(p - alpha_t * m / (v.sqrt() + eps))
                if weight_decay != 0:
                    p.copy_(p - lr * weight_decay * theta_old)

        return loss

    @torch.no_grad()
    def step(self, closure=None):
        """Faster in-place version with equivalent math."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scalar_dtype = (
            torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
        )

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=scalar_dtype)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                step_buf = state["step"]
                step_buf += 1
                t = int(step_buf.item())

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad.square(), alpha=1 - beta2)

                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                alpha_t = lr * (bc2**0.5) / bc1

                theta_old = p.clone()
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                if weight_decay != 0:
                    p.add_(theta_old, alpha=-lr * weight_decay)

        return loss
