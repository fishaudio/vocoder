import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class MMSEncoder(nn.Module):
    def __init__(self, sampling_rate: int = 44100, hop_length: int = 512) -> None:
        super().__init__()

        self.sample_rate = sampling_rate
        self.hop_length = hop_length
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-300m")
        self.model = Wav2Vec2Model.from_pretrained("facebook/mms-300m")

    @torch.no_grad()
    def forward(self, x):
        num_frames = x.shape[-1] // self.hop_length

        x = torchaudio.functional.resample(
            x, orig_freq=self.sample_rate, new_freq=16000
        )
        x = [i.cpu().numpy() for i in x]
        input_values = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).input_values
        input_values = input_values.to(self.model.device)

        x = self.model(input_values).last_hidden_state
        x = x.transpose(1, 2)
        x = torch.functional.F.interpolate(x, size=num_frames, mode="nearest")

        return x
