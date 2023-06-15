import torch
import torchaudio.functional as AF
from torch import Tensor, nn


class RandomHQPitchShift(nn.Module):
    def __init__(
        self,
        probability: float = 1.0,
        pitch_range: int | tuple[int, int] = 12,
        sampling_rate: int = 44100,
    ) -> None:
        super().__init__()

        self.probability = probability

        if isinstance(pitch_range, int):
            pitch_range = (-pitch_range, pitch_range)

        self.pitch_range = pitch_range
        self.sampling_rate = sampling_rate

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        pitch_shift = torch.randint(*self.pitch_range, (1,)).item()
        duration_shift = 2 ** (pitch_shift / 12)

        orig_freq = round(self.sampling_rate * duration_shift)
        orig_freq = orig_freq - (orig_freq % 100)  # avoid creating lots of windows

        y = AF.resample(waveform, orig_freq=orig_freq, new_freq=self.sampling_rate)

        return y
