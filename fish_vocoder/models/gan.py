from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from fish_vocoder.models.vocoder import VocoderModel
from fish_vocoder.modules.losses.stft import MultiResolutionSTFTLoss
from fish_vocoder.utils.grad_norm import grad_norm
from fish_vocoder.utils.mask import sequence_mask


class GANModel(VocoderModel):
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
        multi_resolution_stft_loss: MultiResolutionSTFTLoss,
        num_frames: int,
        crop_length: int | None = None,
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

        # Loss
        self.multi_resolution_stft_loss = multi_resolution_stft_loss

        # Crop length for saving memory
        self.num_frames = num_frames
        self.crop_length = crop_length

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

    def training_generator(self, audio, audio_mask):
        fake_audio, base_loss = self.forward(audio, audio_mask)

        assert fake_audio.shape == audio.shape

        # Apply mask
        audio = audio * audio_mask
        fake_audio = fake_audio * audio_mask

        # Multi-Resolution STFT Loss
        sc_loss, mag_loss = self.multi_resolution_stft_loss(
            fake_audio.squeeze(1), audio.squeeze(1)
        )
        loss_stft = sc_loss + mag_loss

        self.log(
            "train/generator/stft",
            loss_stft,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # L1 Mel-Spectrogram Loss
        # This is not used in backpropagation currently
        audio_mel = self.mel_transforms.loss(audio.squeeze(1))
        fake_audio_mel = self.mel_transforms.loss(fake_audio.squeeze(1))
        loss_mel = F.l1_loss(audio_mel, fake_audio_mel)

        self.log(
            "train/generator/mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Now, we need to reduce the length of the audio to save memory
        if self.crop_length is not None and audio.shape[2] > self.crop_length:
            slice_idx = torch.randint(0, audio.shape[-1] - self.crop_length, (1,))

            audio = audio[..., slice_idx : slice_idx + self.crop_length]
            fake_audio = fake_audio[..., slice_idx : slice_idx + self.crop_length]
            audio_mask = audio_mask[..., slice_idx : slice_idx + self.crop_length]

            assert audio.shape == fake_audio.shape == audio_mask.shape

        # Adv Loss
        loss_adv_all = 0

        for key, disc in self.discriminators.items():
            score_fakes, feat_fake = disc(fake_audio)
            _, feat_real = disc(audio)

            # Adversarial Loss
            loss_fake = 0
            for score_fake in score_fakes:
                loss_fake += torch.mean((1 - score_fake) ** 2)

            # Feature Matching Loss
            loss_fm = 0
            for dr, dg in zip(feat_real, feat_fake):
                for rl, gl in zip(dr, dg):
                    loss_fm += F.l1_loss(rl, gl)

            self.log(
                f"train/generator/adv_{key}",
                loss_fake,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            self.log(
                f"train/generator/adv_fm_{key}",
                loss_fm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_adv_all += loss_fake + loss_fm * 2

        loss_adv_all /= len(self.discriminators)
        loss_gen_all = base_loss + loss_stft * 2.5 + loss_mel * 45 + loss_adv_all

        self.log(
            "train/generator/all",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss_gen_all, audio, fake_audio

    def training_discriminator(self, audio, fake_audio):
        loss_disc_all = 0

        for key, disc in self.discriminators.items():
            scores, _ = disc(audio)
            score_fakes, _ = disc(fake_audio.detach())

            loss_disc = 0

            for score, score_fake in zip(scores, score_fakes):
                loss_disc += torch.mean((score - 1) ** 2) + torch.mean(
                    (score_fake) ** 2
                )

            self.log(
                f"train/discriminator/{key}",
                loss_disc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_disc_all += loss_disc

        loss_disc_all /= len(self.discriminators)

        self.log(
            "train/discriminator/all",
            loss_disc_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss_disc_all

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audio, lengths = batch["audio"], batch["lengths"]
        audio_mask = sequence_mask(lengths)[:, None, :].to(audio.device, torch.float32)

        # Generator
        optim_g.zero_grad()
        loss_gen_all, audio, fake_audio = self.training_generator(audio, audio_mask)
        self.manual_backward(loss_gen_all)

        self.log(
            "train/generator/grad_norm",
            grad_norm(self.generator.parameters()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        optim_g.step()

        # Discriminator
        assert fake_audio.shape == audio.shape

        optim_d.zero_grad()
        loss_disc_all = self.training_discriminator(audio, fake_audio)
        self.manual_backward(loss_disc_all)

        for key, disc in self.discriminators.items():
            self.log(
                f"train/discriminator/grad_norm_{key}",
                grad_norm(disc.parameters()),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        optim_d.step()

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def forward(self, audio, mask=None):
        input_mels = self.mel_transforms.input(audio.squeeze(1))
        fake_audio = self.generator(input_mels)

        return fake_audio, 0

    def validation_step(self, batch: Any, batch_idx: int):
        audio, lengths = batch["audio"], batch["lengths"]
        audio_mask = sequence_mask(lengths)[:, None, :].to(audio.device, torch.float32)

        # Generator
        fake_audio, _ = self.forward(audio, audio_mask)
        assert fake_audio.shape == audio.shape

        # Apply mask
        audio = audio * audio_mask
        fake_audio = fake_audio * audio_mask

        # L1 Mel-Spectrogram Loss
        audio_mel = self.mel_transforms.loss(audio.squeeze(1))
        fake_audio_mel = self.mel_transforms.loss(fake_audio.squeeze(1))
        loss_mel = F.l1_loss(audio_mel, fake_audio_mel)

        self.log(
            "val/metrics/mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Report other metrics
        self.report_val_metrics(fake_audio, audio, lengths)
