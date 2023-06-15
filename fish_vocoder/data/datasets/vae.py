from pathlib import Path
from typing import Callable, Optional

import torch
import torchaudio
import torchaudio.functional as AF
from torch import Tensor
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sampling_rate: int = 44100,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()

        assert Path(root).exists(), f"Path {root} does not exist."
        assert (
            sampling_rate > 0
        ), f"Sampling rate must be positive, got {sampling_rate}."
        assert transform is not None, "transform must be provided."

        self.audio_paths = list(Path(root).rglob("**/*.wav"))
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = self.audio_paths[idx]

        audio, sr = torchaudio.load(audio)
        audio = AF.resample(audio, orig_freq=sr, new_freq=self.sampling_rate)

        # If audio is not mono, convert it to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # You should do random crop and pitch augmentation here.
        audio = self.transform(audio)

        # Do normalization to avoid clipping
        if audio.abs().max() >= 1.0:
            audio /= audio.abs().max() / 0.99

        return {
            "audio": audio,
        }


def collate_fn(batch):
    lengths = [b["audio"].shape[-1] for b in batch]
    max_len = max(lengths)

    for i, b in enumerate(batch):
        pad = max_len - b["audio"].shape[-1]
        batch[i]["audio"] = torch.nn.functional.pad(b["audio"], (0, pad))

    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
