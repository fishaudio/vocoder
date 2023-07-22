from pathlib import Path

import hydra
import lightning as L
import pyrootutils
import torch
import torch.nn.functional as F
import torchmetrics
from hydra.utils import instantiate
from lightning.fabric.loggers import TensorBoardLogger
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary
from tqdm.rich import tqdm

from fish_vocoder.utils.mask import sequence_mask

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

from fish_vocoder.utils.logger import logger
from fish_vocoder.utils.viz import plot_mel


@hydra.main(
    config_path="fish_vocoder/configs", version_base="1.3", config_name="hifigan"
)
def main(cfg: DictConfig):
    logger.add(f"{cfg.paths.run_dir}/train.log")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set seed to {cfg.seed}")

    data: L.LightningDataModule = instantiate(cfg.data)
    logger.info(f"Data: {data}")

    model = instantiate(cfg.model)

    template_len = cfg.base.hop_length * 85
    generator_summary = summary(
        model.generator, [(cfg.base.hidden_size, 85), (1, template_len)], verbose=0
    )
    logger.info(f"Generator summary:\n{generator_summary}")

    mpd_summary = summary(model.mpd, (1, template_len), verbose=0)
    logger.info(f"MPD summary:\n{mpd_summary}")

    mrd_summary = summary(model.mrd, (1, template_len), verbose=0)
    logger.info(f"MRD summary:\n{mrd_summary}")

    optimizer_partial = instantiate(cfg.optimizer)
    logger.info(f"Optimizer: {optimizer_partial}")

    lr_scheduler = instantiate(cfg.lr_scheduler)
    logger.info(f"LR scheduler: {lr_scheduler}")

    fabric: L.Fabric = instantiate(cfg.trainer)
    logger.info(f"Trainer: {fabric}")

    fabric.launch()

    # Now the training begins
    model = fabric.setup_module(model)

    # Instantiate optimizer and LR scheduler here, so that FSDP can work properly
    optim_g = optimizer_partial(params=model.generator.parameters())

    optim_d = optimizer_partial(
        params=[*model.mpd.parameters(), *model.mrd.parameters()]
    )

    optim_g, optim_d = fabric.setup_optimizers(optim_g, optim_d)

    # Get dataloaders
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    global_step = 0
    ckpt_dir = Path(cfg.paths.ckpt_dir)

    if ckpt_dir.exists():
        all_ckpts = natsorted(ckpt_dir.glob("*.ckpt"))
        if len(all_ckpts) != 0:
            logger.info(f"Found {len(all_ckpts)} checkpoints in {cfg.paths.ckpt_dir}")
            remainder = fabric.load(
                all_ckpts[-1], {"model": model, "optim_g": optim_g, "optim_d": optim_d}
            )
            logger.info(f"Loaded checkpoint {all_ckpts[-1]}")
            global_step = remainder["global_step"]

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bar = tqdm(
        desc="Training",
        disable=fabric.global_rank != 0,
        initial=global_step,
        total=cfg.loop.max_steps,
    )

    while global_step < cfg.loop.max_steps:
        for batch in train_dataloader:
            # Validation
            if global_step % cfg.loop.val_interval == 0:
                model.eval()
                bar.set_description("Validating")

                for idx, batch in enumerate(val_dataloader):
                    log_dict = validation_step(
                        cfg=cfg,
                        fabric=fabric,
                        batch=batch,
                        model=model,
                        step=global_step + idx,
                    )
                    fabric.log_dict(log_dict, step=global_step + idx)

            model.train()
            bar.set_description("Training")

            # Set learning rate
            for optim in [optim_g, optim_d]:
                for group in optim.param_groups:
                    group["lr"] = lr_scheduler(global_step)

            log_dict = training_step(
                cfg=cfg,
                fabric=fabric,
                batch=batch,
                model=model,
                optim_g=optim_g,
                optim_d=optim_d,
            )

            # log learning rate
            log_dict["lr/generator"] = optim_g.param_groups[0]["lr"]
            log_dict["lr/discriminator"] = optim_d.param_groups[0]["lr"]

            if global_step % cfg.loop.log_interval == 0:
                fabric.log_dict(log_dict, step=global_step)

            global_step += 1
            bar.update()

            # Save checkpoint
            if global_step % cfg.loop.save_interval == 0:
                save_path = ckpt_dir / f"{global_step}.ckpt"
                fabric.save(
                    save_path,
                    {
                        "model": model,
                        "optim_g": optim_g,
                        "optim_d": optim_d,
                        "global_step": global_step,
                    },
                )

                logger.info(f"Saved checkpoint at {save_path}")


def training_step(
    cfg: DictConfig,
    fabric: L.Fabric,
    batch: dict,
    model: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
) -> dict:
    log_dict = {}

    gt_y, lengths = batch["audio"], batch["lengths"]
    y_mask = sequence_mask(lengths)[:, None, :].to(gt_y.device, torch.float32)

    # Forward
    features = model.encoder(gt_y.squeeze(1))

    # Impossible to generate 1000 frames
    # Only pick 32 frames to decode [16, 1024, 128] -> [16, 1024, 32]
    decode_frames = 32
    frame_idx = torch.randint(0, features.shape[-1] - decode_frames, (1,))
    features = features[..., frame_idx : frame_idx + decode_frames]
    gt_y = gt_y[
        ...,
        frame_idx
        * cfg.base.hop_length : (frame_idx + decode_frames)
        * cfg.base.hop_length,
    ]
    y_mask = y_mask[
        ...,
        frame_idx
        * cfg.base.hop_length : (frame_idx + decode_frames)
        * cfg.base.hop_length,
    ]
    g_hat_y = model.generator(features)

    min_length = min(g_hat_y.shape[-1], gt_y.shape[-1])
    g_hat_y = g_hat_y[..., :min_length]
    gt_y = gt_y[..., :min_length]

    g_hat_y = g_hat_y * y_mask

    # Scale Invariant SNR
    with torch.no_grad():
        si_snr = torchmetrics.functional.scale_invariant_signal_distortion_ratio(
            g_hat_y, gt_y
        ).mean()
        log_dict["train/si_snr"] = si_snr

    # Discriminator Loss
    optim_d.zero_grad()

    # MPD Loss
    y_g_hat_x, _ = model.mpd(g_hat_y.detach())
    y_x, _ = model.mpd(gt_y)
    loss_mpd = discriminator_loss(y_x, y_g_hat_x)
    log_dict["train/loss_mpd_d"] = loss_mpd

    # MRD Loss
    y_g_hat_x, _ = model.mrd(g_hat_y.detach())
    y_x, _ = model.mrd(gt_y)
    loss_mrd = discriminator_loss(y_x, y_g_hat_x)
    log_dict["train/loss_mrd_d"] = loss_mrd

    loss_d = loss_mpd + loss_mrd
    log_dict["train/loss_d"] = loss_d

    fabric.backward(loss_d)
    optim_d.step()

    # Generator Loss
    optim_g.zero_grad()

    # L1 Mel Loss
    loss_mel = generator_mel_loss(gt_y, g_hat_y, model.multi_scale_mel_transforms)
    log_dict["train/loss_mel"] = loss_mel

    # MPD Loss
    y_g_hat_x, y_hat_mpd_fmap = model.mpd(g_hat_y)
    _, y_mpd_fmap = model.mpd(gt_y)
    loss_mpd = generator_adv_loss(y_g_hat_x)
    log_dict["train/loss_mpd_g"] = loss_mpd

    loss_mpd_fm = feature_matching_loss(y_mpd_fmap, y_hat_mpd_fmap)
    log_dict["train/loss_mpd_fm"] = loss_mpd_fm

    # MRD Loss
    y_g_hat_x, y_hat_mrd_fmap = model.mrd(g_hat_y)
    _, y_mrd_fmap = model.mrd(gt_y)
    loss_mrd = generator_adv_loss(y_g_hat_x)
    log_dict["train/loss_mrd_g"] = loss_mrd

    loss_mrd_fm = feature_matching_loss(y_mrd_fmap, y_hat_mrd_fmap)
    log_dict["train/loss_mrd_fm"] = loss_mrd_fm

    # All generator Loss
    loss_g = loss_mel * 45 + loss_mpd + loss_mrd + loss_mpd_fm + loss_mrd_fm
    log_dict["train/loss_g"] = loss_g

    fabric.backward(loss_g)
    optim_g.step()

    return log_dict


@torch.no_grad()
def validation_step(
    cfg: DictConfig,
    fabric: L.Fabric,
    batch: dict,
    model: torch.nn.Module,
    step: int,
) -> dict:
    log_dict = {}

    gt_y, lengths = batch["audio"], batch["lengths"]
    y_mask = sequence_mask(lengths)[:, None, :].to(gt_y.device, torch.float32)
    spec_lengths = lengths // cfg.base.hop_length

    # Forward
    features = model.encoder(gt_y.squeeze(1))
    g_hat_y = model.generator(features)

    min_length = min(g_hat_y.shape[-1], gt_y.shape[-1])
    g_hat_y = g_hat_y[..., :min_length]
    gt_y = gt_y[..., :min_length]

    g_hat_y = g_hat_y * y_mask

    # Scale Invariant SNR
    with torch.no_grad():
        si_snr = torchmetrics.functional.scale_invariant_signal_distortion_ratio(
            g_hat_y, gt_y
        ).mean()
        log_dict["val/si_snr"] = si_snr

    # Mel Loss
    loss_mel = generator_mel_loss(gt_y, g_hat_y, model.multi_scale_mel_transforms)
    log_dict["val/loss_mel"] = loss_mel

    # Log Audio and Mel Spectrograms
    tb_loggers = [
        logger for logger in fabric.loggers if isinstance(logger, TensorBoardLogger)
    ]
    if len(tb_loggers) == 0:
        return log_dict

    gt_mel_spec = model.viz_mel_transform(gt_y.squeeze(1))
    gen_mel_spec = model.viz_mel_transform(g_hat_y.squeeze(1))

    for tb_logger in tb_loggers:
        for idx, (y, y_hat, y_length, gt_mel, gen_mel, mel_len) in enumerate(
            zip(gt_y, g_hat_y, lengths, gt_mel_spec, gen_mel_spec, spec_lengths)
        ):
            tb_logger.experiment.add_audio(
                f"audio-{idx}/target",
                y[..., :y_length],
                step,
                cfg.base.sampling_rate,
            )

            tb_logger.experiment.add_audio(
                f"audio-{idx}/generated",
                y_hat[..., :y_length],
                step,
                cfg.base.sampling_rate,
            )

            mel_fig = plot_mel(
                [gt_mel[..., :mel_len], gen_mel[..., :mel_len]],
                ["Ground Truth", "Generated"],
            )

            tb_logger.experiment.add_figure(
                f"mels/{idx}",
                mel_fig,
                step,
                close=True,
            )

    return log_dict


def generator_mel_loss(y, y_hat, multi_scale_mel_transforms):
    loss_mel = []

    for mel_transform in multi_scale_mel_transforms:
        y_mel = mel_transform(y.squeeze(1))
        y_g_hat_mel = mel_transform(y_hat.squeeze(1))
        loss_mel.append(F.l1_loss(y_mel, y_g_hat_mel))

    return sum(loss_mel) / len(loss_mel)


def generator_adv_loss(disc_outputs):
    losses = []

    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        losses.append(l)

    return sum(losses) / len(losses)


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        losses.append((r_loss + g_loss) / 2)

    return sum(losses) / len(losses)


def feature_matching_loss(disc_real_outputs, disc_generated_outputs):
    losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        for rl, gl in zip(dr, dg):
            losses.append(F.l1_loss(rl, gl))

    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
