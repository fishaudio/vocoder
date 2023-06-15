# from pathlib import Path

# import librosa
# import numpy as np
# import soundfile as sf
# import torch
# import torchmetrics

import soundfile as sf
from torch import nn

from fish_vocoder.data.datasets.refine import AudioRefineDataset
from fish_vocoder.data.transforms.discontinuous import RandomDiscontinuous
from fish_vocoder.data.transforms.hq_pitch_shift import RandomHQPitchShift
from fish_vocoder.data.transforms.pad_crop import RandomPadCrop

for i in AudioRefineDataset(
    "dataset/train",
    base_transform=nn.Sequential(
        RandomHQPitchShift(probability=1, sampling_rate=44100),
        RandomPadCrop(probability=1, crop_length=44100 * 3, padding=True),
    ),
    augment_transform=nn.Sequential(
        RandomDiscontinuous(probability=1, sampling_rate=44100),
    ),
):
    sf.write("test.wav", i["source"].T, 44100)
    sf.write("test2.wav", i["target"].T, 44100)

    input("Press Enter to continue...")

# from dataset import RandomHQPitchShift
# from fish_vocoder.data.transforms.discontinuous import RandomDiscontinuous

# audio, sr = librosa.load("dataset/train/M4Singer/Alto-1#漫长的告白/0000.wav", sr=44100)
# print(audio.shape)
# raw_audio = audio.copy()

# # 随机断音
# # fix random seed for reproducibility
# # np.random.seed(0)
# # for _ in range(2):
# #     start = np.random.randint(0, len(audio) - 512)
# #     end = start + np.random.randint(0, 512)
# #     audio[start:end] = 0

# raw_audio = torch.from_numpy(raw_audio)
# audio = torch.from_numpy(audio)

# audio = RandomDiscontinuous(probability=1, sampling_rate=44100)(raw_audio)
# print(torchmetrics.functional.scale_invariant_signal_distortion_ratio(audio, raw_audio))

# # write to file
# sf.write("test.wav", audio, sr)
