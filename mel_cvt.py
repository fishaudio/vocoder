import torch

data = torch.load("other/一半一半.mel.pt", map_location="cpu")
all_mels = [i["mel"] for i in data]
all_mel = torch.cat(all_mels, dim=1) / 0.434294

# all_mel = torch.zeros((1, int(data[-1]['offset'] * 44100 / 512) + 500, 128))

# for seg in data:
#     offset = int(seg['offset'] * 44100 / 512)
#     mel = seg["mel"]

#     all_mel[:, offset:offset + mel.shape[1], :] = mel

print(all_mel.shape)

torch.save(all_mel, "other/一半一半.1.mel.pt")
