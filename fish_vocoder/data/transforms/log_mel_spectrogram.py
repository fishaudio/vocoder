import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram as _MelSpectrogram


class LogMelSpectrogram(_MelSpectrogram):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=True,
        power=1.0,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
        *args,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            power=power,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
            *args,
            **kwargs,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.compress(x)

        return x
