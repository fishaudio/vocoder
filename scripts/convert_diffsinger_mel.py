import torch

data = torch.load("能解答一切的答案.mel.pt", map_location="cpu")
all_mels = [i["mel"] for i in data]
# all_mel = torch.cat(all_mels, dim=1) / 0.434294

all_mel = (
    torch.zeros(
        (1, int(data[-1]["offset"] * 44100 / 512) + data[-1]["mel"].shape[1], 128)
    )
    - 11.512925
)

for seg in data:
    offset = int(seg["offset"] * 44100 / 512)
    mel = seg["mel"] / 0.434294
    all_mel[:, offset : offset + mel.shape[1], :] = mel

torch.save(all_mel, "other/mels/能解答一切的答案.mel.pt")
