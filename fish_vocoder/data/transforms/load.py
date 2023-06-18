import torchaudio
from torch import Tensor, nn
from torchaudio import functional as AF


class LoadAudio(nn.Module):
    def __init__(self, sampling_rate: int = 44100, to_mono: bool = True):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.to_mono = to_mono

    def forward(self, audio_path: str) -> Tensor:
        audio, sr = torchaudio.load(audio_path)
        audio = AF.resample(audio, orig_freq=sr, new_freq=self.sampling_rate)

        # If audio is not mono, convert it to mono
        if self.to_mono and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio
