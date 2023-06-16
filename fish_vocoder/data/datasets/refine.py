from pathlib import Path
from typing import Callable, Optional

import torch
import torchaudio
import torchaudio.functional as AF
from torch import Tensor
from torch.utils.data import Dataset

from fish_vocoder.utils.file import AUDIO_EXTENSIONS, list_files


class AudioRefineDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sampling_rate: int = 44100,
        base_transform: Optional[Callable[[Tensor], Tensor]] = None,
        augment_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()

        assert Path(root).exists(), f"Path {root} does not exist."
        assert (
            sampling_rate > 0
        ), f"Sampling rate must be positive, got {sampling_rate}."
        assert base_transform is not None, "Base transform must be provided."
        assert augment_transform is not None, "Augment transform must be provided."

        self.audio_paths = list_files(root, AUDIO_EXTENSIONS, recursive=True)
        self.sampling_rate = sampling_rate
        self.base_transform = base_transform
        self.augment_transform = augment_transform

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
        target_audio = self.base_transform(audio)

        # Do normalization to avoid clipping
        if target_audio.abs().max() >= 1.0:
            target_audio /= target_audio.abs().max() / 0.99

        # Then add discontinuous / noise here
        # Note: need to copy the tensor to avoid inplace operation
        source_audio = self.augment_transform(target_audio.clone())

        assert source_audio.shape == target_audio.shape

        return {
            "source": source_audio,
            "target": target_audio,
        }


def collate_fn(batch):
    lengths = [b["source"].shape[-1] for b in batch]
    max_len = max(lengths)

    for i, b in enumerate(batch):
        pad = max_len - b["source"].shape[-1]
        batch[i]["source"] = torch.nn.functional.pad(b["source"], (0, pad))
        batch[i]["target"] = torch.nn.functional.pad(b["target"], (0, pad))

    return {
        "source": torch.stack([b["source"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
