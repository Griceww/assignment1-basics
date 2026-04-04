import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_possible_starts = len(dataset) - context_length
    starts = np.random.randint(0, num_possible_starts, size=batch_size)
    offsets = np.arange(context_length)

    x_np = dataset[starts[:, None] + offsets[None, :]]
    y_np = dataset[starts[:, None] + offsets[None, :] + 1]

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
