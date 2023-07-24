from torch import nn

from fish_vocoder.data.transforms.spectrogram import LogMelSpectrogram
from fish_vocoder.models.vocoder import VocoderModel


class HiFiGANModel(VocoderModel):
    def __init__(
        self,
        generator: nn.Module,
        discriminators: dict[str, nn.Module],
        mel_spec_extractor: LogMelSpectrogram,
    ):
        pass
