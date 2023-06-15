import torch
from torch import Tensor, nn


class RandomCrop(nn.Module):
    def __init__(
        self,
        probability: float = 1.0,
        crop_length: int = 44100 * 3,
    ) -> None:
        super().__init__()

        self.probability = probability
        self.crop_length = crop_length

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        if waveform.shape[-1] <= self.crop_length:
            return waveform

        start_idx = torch.randint(0, waveform.shape[-1] - self.crop_length, (1,)).item()
        end_idx = start_idx + self.crop_length

        return waveform[..., start_idx:end_idx]
