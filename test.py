import hydra
import pyrootutils
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
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
    model.load_state_dict(torch.load("other/step_000340000.ckpt")["state_dict"])
    model.eval()
    model.cuda()

    model.generator.remove_weight_norm()

    gt_y, sr = torchaudio.load("other/直升机.wav")
    gt_y = torchaudio.functional.resample(gt_y[0], sr, 44100)
    # pitch shift
    # gt_y = torchaudio.functional.pitch_shift(gt_y, sr, -8)

    gt_y = gt_y[None, None, ...].to(model.device, torch.float32)
    lengths = torch.IntTensor([gt_y.shape[-1]])
    # pad to 512
    gt_y = F.pad(gt_y, (0, 512 - (512 % gt_y.shape[-1])))
    logger.info(f"gt_y shape: {gt_y.shape}, lengths: {lengths}")

    input_mels = model.mel_transforms.input(gt_y.squeeze(1))
    fake_audio = model.generator(input_mels)

    sf.write("g_hat_y.wav", fake_audio[0].cpu().numpy().T, 44100)


if __name__ == "__main__":
    main()
