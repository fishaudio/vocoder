import torch
from torch import Tensor, nn


class RandomLoudness(nn.Module):
    def __init__(
        self, probability: float = 1.0, loudness_range: tuple[int, int] = (0.1, 0.9)
    ) -> None:
        super().__init__()

        self.probability = probability
        self.loudness_range = loudness_range

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        new_loudness = (
            torch.rand(1).item() * (self.loudness_range[1] - self.loudness_range[0])
            + self.loudness_range[0]
        )
        max_loudness = torch.max(torch.abs(waveform))
        waveform = waveform * (new_loudness / (max_loudness + 1e-5))

        return waveform


class LoudnessNorm(nn.Module):
    def __init__(self, probability: float = 1.0) -> None:
        super().__init__()

        self.probability = probability

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        max_loudness = torch.max(torch.abs(waveform))
        waveform = waveform / (max_loudness + 1e-5)

        return waveform
