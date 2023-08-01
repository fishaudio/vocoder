from collections import defaultdict
from pathlib import Path

import click
import librosa
import numpy as np
import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from tqdm import tqdm


def pesq_nb(target, preds, sr):
    target = librosa.resample(target, orig_sr=sr, target_sr=8000)
    preds = librosa.resample(preds, orig_sr=sr, target_sr=8000)
    target = torch.from_numpy(target)
    preds = torch.from_numpy(preds)

    return perceptual_evaluation_speech_quality(preds, target, 8000, "nb").item()


def pesq_wb(target, preds, sr):
    target = librosa.resample(target, orig_sr=sr, target_sr=16000)
    preds = librosa.resample(preds, orig_sr=sr, target_sr=16000)
    target = torch.from_numpy(target)
    preds = torch.from_numpy(preds)

    return perceptual_evaluation_speech_quality(preds, target, 16000, "wb").item()


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument(
    "generated", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
def main(source, generated):
    source = Path(source)
    generated = Path(generated)

    assert source.is_dir()
    assert generated.is_dir()

    source_files = sorted(list(source.rglob("*.wav")))
    scores = defaultdict(list)
    bar = tqdm(source_files)

    for idx, source_file in enumerate(tqdm(source_files)):
        generated_file = generated / source_file.relative_to(source)
        if not generated_file.exists():
            print(f"{generated_file} does not exist")
            continue

        source_audio, sr = librosa.load(source_file, sr=44100)
        generated_audio, _ = librosa.load(generated_file, sr=44100)

        min_len = min(len(source_audio), len(generated_audio))
        assert max(len(source_audio) - min_len, len(generated_audio) - min_len) < 1000

        source_audio = source_audio[:min_len]
        generated_audio = generated_audio[:min_len]

        try:
            scores["pesq_nb"].append(pesq_nb(source_audio, generated_audio, sr))
            scores["pesq_wb"].append(pesq_wb(source_audio, generated_audio, sr))
        except Exception:
            print(f"Error processing {source_file}")
            continue

        if idx % 10 == 0:
            all_metrics = [f"{k}: {np.mean(v):.2f}" for k, v in scores.items()]
            bar.set_description(", ".join(all_metrics))

    print("Average scores:")
    for k, v in scores.items():
        print(f"    {k}: {np.mean(v):.2f}")


if __name__ == "__main__":
    main()
