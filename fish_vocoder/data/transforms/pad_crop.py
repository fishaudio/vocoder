import torch
from torch import Tensor, nn


class RandomPadCrop(nn.Module):
    def __init__(
        self,
        probability: float = 1.0,
        crop_length: int = 44100 * 3,
        padding: bool = True,
    ) -> None:
        super().__init__()

        self.probability = probability
        self.crop_length = crop_length
        self.padding = padding

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        if self.padding and waveform.shape[-1] < self.crop_length:
            pad_length = self.crop_length - waveform.shape[-1]
            pad_left = torch.randint(0, pad_length, (1,)).item()
            pad_right = pad_length - pad_left

            return torch.nn.functional.pad(
                waveform, (pad_left, pad_right), mode="constant", value=0
            )

        start_idx = torch.randint(0, waveform.shape[-1] - self.crop_length, (1,)).item()
        end_idx = start_idx + self.crop_length

        return waveform[..., start_idx:end_idx]
