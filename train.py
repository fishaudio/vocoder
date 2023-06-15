import hydra
import lightning as L
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary

# Allow TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

from fish_vocoder.utils.logger import logger


@hydra.main(config_path="fish_vocoder/configs", version_base="1.3")
def main(cfg: DictConfig):
    logger.add(f"{cfg.paths.run_dir}/train.log")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set seed to {cfg.seed}")

    data = instantiate(cfg.data)
    logger.info(f"Data: {data}")

    model = instantiate(cfg.model)
    generator_summary = summary(model.generator, [(128, 85), (1, 43520)], verbose=0)
    logger.info(f"Generator summary:\n{generator_summary}")

    mpd_summary = summary(model.mpd, (1, 43520), verbose=0)
    logger.info(f"MPD summary:\n{mpd_summary}")

    mrd_summary = summary(model.mrd, (1, 43520), verbose=0)
    logger.info(f"MRD summary:\n{mrd_summary}")

    optimizer_partial = instantiate(cfg.optimizer)
    logger.info(f"Optimizer: {optimizer_partial}")

    lr_scheduler_partial = instantiate(cfg.lr_scheduler)
    logger.info(f"LR scheduler: {lr_scheduler_partial}")

    fabric: L.Fabric = instantiate(cfg.trainer)
    logger.info(f"Trainer: {fabric}")

    fabric.launch()

    # Now the training begins
    model = fabric.setup_module(model)

    # Instantiate optimizer and LR scheduler here, so that FSDP can work properly
    generator_optimizer = optimizer_partial(params=model.generator.parameters())
    generator_lr_scheduler = lr_scheduler_partial(optimizer=generator_optimizer)

    discriminator_optimizer = optimizer_partial(
        params=[*model.mpd.parameters(), *model.mrd.parameters()]
    )
    discriminator_lr_scheduler = lr_scheduler_partial(optimizer=discriminator_optimizer)

    generator_optimizer, discriminator_optimizer = fabric.setup_optimizers(
        generator_optimizer, discriminator_optimizer
    )

    # Begin training


if __name__ == "__main__":
    main()
