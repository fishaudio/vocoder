import torch
from torch import Tensor, nn


class RandomDiscontinuous(nn.Module):
    def __init__(
        self,
        probability: float = 1.0,
        silent_range: tuple[float, float] = (0.01, 0.1),
        silent_ratio_range: tuple[float, float] = (0.1, 0.2),
        sampling_rate: int = 44100,
    ) -> None:
        super().__init__()

        self.probability = probability
        self.sampling_rate = sampling_rate
        self.silent_range = (
            int(silent_range[0] * sampling_rate),
            int(silent_range[1] * sampling_rate),
        )
        self.silent_ratio_range = silent_ratio_range

    def forward(self, waveform: Tensor) -> Tensor:
        if torch.rand(1) > self.probability:
            return waveform

        current_silent_length = 0
        total_silent_length = torch.randint(
            int(self.silent_ratio_range[0] * waveform.shape[-1]),
            int(self.silent_ratio_range[1] * waveform.shape[-1]),
            (1,),
        ).item()

        while current_silent_length < total_silent_length:
            silent_length = torch.randint(*self.silent_range, (1,)).item()
            start_idx = torch.randint(
                0, waveform.shape[-1] - silent_length, (1,)
            ).item()
            end_idx = start_idx + silent_length
            current_silent_length += silent_length

            # 0: all silent, 1: linear fade in and out
            silent_mode = torch.randint(0, 2, (1,)).item()

            if silent_mode == 0:
                waveform[..., start_idx:end_idx] = 0
            elif silent_mode == 1:
                waveform[..., start_idx:end_idx] *= torch.cat(
                    (
                        torch.linspace(0, 1, silent_length // 2),
                        torch.linspace(1, 0, silent_length - silent_length // 2),
                    )
                )

        return waveform
