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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

from fish_vocoder.utils.logger import logger
from fish_vocoder.utils.viz import plot_mel


@hydra.main(config_path="fish_vocoder/configs", version_base="1.3", config_name="vae")
def main(cfg: DictConfig):
    logger.add(f"{cfg.paths.run_dir}/train.log")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set seed to {cfg.seed}")

    data: L.LightningDataModule = instantiate(cfg.data)
    logger.info(f"Data: {data}")

    model = instantiate(cfg.model)

    posterior_encoder_summary = summary(
        model.posterior_encoder,
        (cfg.model.modules.posterior_encoder.in_channels, 85),
        verbose=0,
    )
    logger.info(f"Posterior encoder summary:\n{posterior_encoder_summary}")

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

    lr_scheduler_partial = instantiate(cfg.lr_scheduler)
    logger.info(f"LR scheduler: {lr_scheduler_partial}")

    fabric: L.Fabric = instantiate(cfg.trainer)
    logger.info(f"Trainer: {fabric}")

    fabric.launch()

    # Now the training begins
    model = fabric.setup_module(model)

    # Instantiate optimizer and LR scheduler here, so that FSDP can work properly
    optim_g = optimizer_partial(params=model.generator.parameters())
    generator_lr_scheduler = lr_scheduler_partial(optimizer=optim_g)

    optim_d = optimizer_partial(
        params=[*model.mpd.parameters(), *model.mrd.parameters()]
    )
    discriminator_lr_scheduler = lr_scheduler_partial(optimizer=optim_d)

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
        # log learning rate
        log_dict = {
            "lr/generator": optim_g.param_groups[0]["lr"],
            "lr/discriminator": optim_d.param_groups[0]["lr"],
        }
        fabric.log_dict(log_dict, step=global_step)

        model.train()

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

            bar.set_description("Training")
            log_dict = training_step(
                cfg=cfg,
                fabric=fabric,
                batch=batch,
                model=model,
                optim_g=optim_g,
                optim_d=optim_d,
            )

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

        # Update LR schedulers
        generator_lr_scheduler.step()
        discriminator_lr_scheduler.step()


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
    y_mask = sequence_mask(lengths)[:, None, :].to(gt_y.device)

    # Forward VAE
    with torch.no_grad():
        gt_spec = model.spectrogram_transform(gt_y.squeeze(1))
        gt_spec = gt_spec[..., : lengths.max() // cfg.base.hop_length]

    spec_lengths = lengths // cfg.base.hop_length
    z, mean, std, z_mask = model.posterior_encoder(gt_spec, spec_lengths)
    g_hat_y = model.generator(z)

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
    log_dict["train/loss_mpd"] = loss_mpd

    # MRD Loss
    y_g_hat_x, _ = model.mrd(g_hat_y.detach())
    y_x, _ = model.mrd(gt_y)
    loss_mrd = discriminator_loss(y_x, y_g_hat_x)
    log_dict["train/loss_mrd"] = loss_mrd

    loss_d = loss_mpd + loss_mrd
    log_dict["train/loss_d"] = loss_d

    fabric.backward(loss_d)
    optim_d.step()

    # Generator Loss
    optim_g.zero_grad()

    # L1 Mel Loss
    loss_mel = generator_mel_loss(gt_y, g_hat_y, model.multi_scale_mel_transforms)
    log_dict["train/loss_mel"] = loss_mel

    # L1 Envelope Loss
    loss_envelope = generator_envelope_loss(gt_y, g_hat_y)
    log_dict["train/loss_envelope"] = loss_envelope

    # MPD Loss
    y_g_hat_x, _ = model.mpd(g_hat_y)
    loss_mpd = generator_adv_loss(y_g_hat_x)
    log_dict["train/loss_mpd"] = loss_mpd

    # MRD Loss
    y_g_hat_x, _ = model.mrd(g_hat_y)
    loss_mrd = generator_adv_loss(y_g_hat_x)
    log_dict["train/loss_mrd"] = loss_mrd

    # KL Loss
    loss_kl = kl_loss(mean, std, z_mask)
    log_dict["train/loss_kl"] = loss_kl

    # All generator Loss
    loss_g = loss_mel * 45 + loss_envelope + loss_mpd + loss_mrd + loss_kl
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
    y_mask = sequence_mask(lengths)[:, None, :].to(gt_y.device)

    # Forward VAE
    gt_spec = model.spectrogram_transform(gt_y.squeeze(1))
    gt_spec = gt_spec[..., : lengths.max() // cfg.base.hop_length]

    spec_lengths = lengths // cfg.base.hop_length
    z, mean, std, z_mask = model.posterior_encoder(gt_spec, spec_lengths)
    g_hat_y = model.generator(z)

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

    # KL Loss
    loss_kl = kl_loss(mean, std, z_mask)
    log_dict["val/loss_kl"] = loss_kl

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
        y_mel = mel_transform(y)
        y_g_hat_mel = mel_transform(y_hat)
        loss_mel.append(F.l1_loss(y_mel, y_g_hat_mel))

    return sum(loss_mel) / len(loss_mel)


def generator_envelope_loss(y, y_hat):
    def extract_envelope(signal, kernel_size=512, stride=256):
        envelope = F.max_pool1d(signal, kernel_size=kernel_size, stride=stride)
        return envelope

    y_envelope = extract_envelope(y)
    y_hat_envelope = extract_envelope(y_hat)

    y_reverse_envelope = extract_envelope(-y)
    y_hat_reverse_envelope = extract_envelope(-y_hat)

    loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
        y_reverse_envelope, y_hat_reverse_envelope
    )

    return loss_envelope


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


@torch.autocast("cuda", enabled=False)
def kl_loss(mean, std, mask):
    losses = 0.5 * (mean**2 + std**2 - torch.log(std**2) - 1)

    if torch.isinf(torch.masked_select(losses, mask.to(bool)).mean()):
        print(
            mean.shape,
            std.shape,
            losses.shape,
            torch.max(mean),
            torch.max(std),
            torch.max(mean**2),
            torch.max(std**2),
            torch.max(torch.log(std**2)),
            torch.masked_select(losses, mask.to(bool)).mean()
        )
    return torch.masked_select(losses, mask.to(bool)).mean()


if __name__ == "__main__":
    main()
