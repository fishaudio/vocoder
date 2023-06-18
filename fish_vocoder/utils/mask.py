import torch


def sequence_mask(lengths, max_length=None) -> torch.Tensor:
    if max_length is None:
        max_length = lengths.max()

    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    return x.unsqueeze(0) < lengths.unsqueeze(1)
