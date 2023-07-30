import time
from pathlib import Path

import hydra
import librosa
import pyrootutils
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

# flake8: noqa: E402
from fish_vocoder.utils.logger import logger


@hydra.main(config_path="fish_vocoder/configs", version_base="1.3", config_name="train")
@torch.no_grad()
def main(cfg: DictConfig):
    model: LightningModule = instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.cuda()

    if hasattr(model.generator, "remove_weight_norm"):
        model.generator.remove_weight_norm()

    input_path = Path(cfg.input_path)

    if input_path.suffix in [".wav", ".flac", ".mp3"]:
        gt_y, sr = librosa.load(input_path, sr=cfg.model.sampling_rate, mono=False)

        # If mono, add a channel dimension
        if len(gt_y.shape) == 1:
            gt_y = gt_y[None, :]

        # If we have more than one channel, switch to batched mode
        if cfg.pitch_shift != 0:
            gt_y = librosa.effects.pitch_shift(gt_y, sr=sr, n_steps=cfg.pitch_shift)

        gt_y = torch.from_numpy(gt_y)[:, None].to(model.device, torch.float32)
        lengths = torch.IntTensor([gt_y.shape[-1]])
        gt_y = F.pad(
            gt_y, (0, cfg.model.hop_length - (cfg.model.hop_length % gt_y.shape[-1]))
        )
        logger.info(f"gt_y shape: {gt_y.shape}, lengths: {lengths}")
        input_mels = model.mel_transforms.input(gt_y.squeeze(1))
    elif input_path.suffix in [".pt", ".pth"]:
        input_mels = torch.load(input_path, map_location=model.device).to(torch.float32)

        if len(input_mels.shape) == 2:
            input_mels = input_mels[None, ...]

        if input_mels.shape[-1] == cfg.model.num_mels:
            input_mels = input_mels.transpose(1, 2)
    else:
        raise ValueError(f"Unsupported input file type: {input_path.suffix}")

    start = time.time()
    fake_audio = model.generator(input_mels)
    logger.info(f"Time taken: {time.time() - start:.2f}s")

    generated_name = (
        f"generated_{cfg.task_name}_{ckpt['global_step']}_"
        + f"{Path(cfg.input_path).stem}.wav"
    )
    output_path = cfg.output_path or generated_name
    fake_audio = fake_audio.squeeze(1)

    sf.write(output_path, fake_audio.cpu().numpy().T, cfg.model.sampling_rate)
    logger.info(f"Saved generated audio to {output_path}")


if __name__ == "__main__":
    main()
