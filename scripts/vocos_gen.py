from pathlib import Path

import torchaudio
from tqdm import tqdm
from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
source = Path("dataset/LibriTTS/test-other")
target = Path("results/LibriTTS/test-other/vocos-official")

for i in tqdm(list(source.rglob("*.wav"))):
    y, sr = torchaudio.load(i)
    if y.size(0) > 1:  # mix to mono
        y = y.mean(dim=0, keepdim=True)
    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
    y_hat = vocos(y)

    target_file = target / i.relative_to(source)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(target_file, y_hat, sample_rate=24000)
