from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from fish_vocoder.utils.file import AUDIO_EXTENSIONS, list_files


class VocoderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()

        assert Path(root).exists(), f"Path {root} does not exist."
        assert transform is not None, "transform must be provided."

        root = Path(root)

        if root.is_dir():
            self.audio_paths = list_files(root, AUDIO_EXTENSIONS, recursive=True)
        else:
            self.audio_paths = root.read_text().splitlines()

        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = self.audio_paths[idx]
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
