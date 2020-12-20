import torch
import numpy as np


def collate_fn(batch):
    def _convert(x):
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x, dim=0)
        elif isinstance(x[0], (np.ndarray, list, tuple, int, float)):
            return torch.tensor(x)
        elif x[0] is None:
            return None
    batch = list(map(list, zip(*batch)))
    batch = tuple(map(_convert, batch))
    return batch