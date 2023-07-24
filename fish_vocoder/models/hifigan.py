from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from fish_vocoder.models.vocoder import VocoderModel
from fish_vocoder.utils.mask import sequence_mask


class HiFiGANModel(VocoderModel):
    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_mels: int,
        optimizer: Callable,
        lr_scheduler: Callable,
        mel_transforms: nn.ModuleDict,
        generator: nn.Module,
        discriminators: nn.ModuleDict,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            num_mels=num_mels,
        )

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Spectrogram transforms
        self.mel_transforms = mel_transforms

        # Generator and discriminators
        self.generator = generator
        self.discriminators = discriminators

        # Disable automatic optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())
        optimizer_discriminator = self.optimizer_builder(
            self.discriminators.parameters()
        )

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)
        lr_scheduler_discriminator = self.lr_scheduler_builder(optimizer_discriminator)

        return [optimizer_generator, optimizer_discriminator], [
            {
                "scheduler": lr_scheduler_generator,
                "interval": "step",
            },
            {
                "scheduler": lr_scheduler_discriminator,
                "interval": "step",
            },
        ]

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        y, lengths = batch["audio"], batch["lengths"]
        y_mask = sequence_mask(lengths)[:, None, :].to(y.device, torch.float32)
        input_mels = self.mel_transforms.input(y.squeeze(1))
        y_g_hat = self.generator(input_mels)

        assert y_g_hat.shape == y.shape

        # Apply mask
        y = y * y_mask
        y_g_hat = y_g_hat * y_mask

        # Discriminator
        loss_disc_all = 0
        for key, disc in self.discriminators.items():
            y_hat_r, _ = disc(y)
            y_hat_g, _ = disc(y_g_hat.detach())
            loss_disc, _, _ = self.discriminator_loss(y_hat_r, y_hat_g)

            self.log(
                f"train/losses/disc_{key}",
                loss_disc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_disc_all += loss_disc

        self.manual_backward(loss_disc_all)
        optim_d.step()

        self.log(
            "train/losses/disc",
            loss_disc_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Generator
        optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        y_mel = self.mel_transforms.loss(y.squeeze(1))
        y_g_hat_mel = self.mel_transforms.loss(y_g_hat.squeeze(1))
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel)

        self.log(
            "train/losses/mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Adv Loss
        loss_adv_all = 0

        for key, disc in self.discriminators.items():
            _, fmap_r = disc(y)
            y_d_hat_g, fmap_g = disc(y_g_hat)

            loss_fm = self.feature_matching_loss(fmap_r, fmap_g)
            loss_gen, _ = self.generator_loss(y_d_hat_g)

            self.log(
                f"train/losses/gen_{key}",
                loss_gen,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            self.log(
                f"train/losses/fm_{key}",
                loss_fm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_adv_all += loss_gen + loss_fm

        loss_gen_all = loss_mel * 45 + loss_adv_all

        self.manual_backward(loss_gen_all)
        optim_g.step()

        self.log(
            "train/losses/gen",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

        # Report other metrics
        self.report_train_metrics(y_g_hat, y, lengths)

    def validation_step(self, batch: Any, batch_idx: int):
        y, lengths = batch["audio"], batch["lengths"]
        y_mask = sequence_mask(lengths)[:, None, :].to(y.device, torch.float32)
        input_mels = self.mel_transforms.input(y.squeeze(1))
        y_g_hat = self.generator(input_mels)

        assert y_g_hat.shape == y.shape

        # Apply mask
        y = y * y_mask
        y_g_hat = y_g_hat * y_mask

        # L1 Mel-Spectrogram Loss
        y_mel = self.mel_transforms.loss(y.squeeze(1))
        y_g_hat_mel = self.mel_transforms.loss(y_g_hat.squeeze(1))
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel)

        self.log(
            "val/losses/mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Report other metrics
        self.report_val_metrics(y_g_hat, y, lengths)

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []

        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)

            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        losses = []

        for dg in disc_outputs:
            temp = torch.mean((1 - dg) ** 2)
            losses.append(temp)
            loss += temp

        return loss, losses

    @staticmethod
    def feature_matching_loss(disc_real_outputs, disc_generated_outputs):
        losses = []

        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            for rl, gl in zip(dr, dg):
                losses.append(F.l1_loss(rl, gl))

        return sum(losses)
