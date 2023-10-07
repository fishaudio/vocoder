import torch
from encodec.quantization.core_vq import ResidualVectorQuantization

from fish_vocoder.models.gan import GANModel


class VAEModel(GANModel):
    def __init__(self, latent_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_size = latent_size

    def forward(self, audio, mask):
        input_spec = self.mel_transforms.input(audio.squeeze(1))

        latent = self.generator.encoder(input_spec)
        mean, logvar = torch.chunk(latent, 2, dim=1)
        z = self.reparameterize(mean, logvar)
        fake_audio = self.generator.decoder(z)

        kl_loss = self.kl_loss(mean, logvar)

        self.log(
            "train/generator/kl",
            kl_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return fake_audio, kl_loss

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return mean + eps * std

        return mean

    @staticmethod
    def kl_loss(mean, logvar):
        # B, D, T -> B, 1, T
        losses = 0.5 * (mean**2 + torch.exp(logvar) - logvar - 1)
        return losses.mean()


class VQVAEModel(GANModel):
    def __init__(self, latent_size: int, codebook_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_size = latent_size
        self.codebook_size = codebook_size
        self.vq = ResidualVectorQuantization(
            dim=latent_size,
            codebook_size=codebook_size,
            kmeans_init=False,
            num_quantizers=2,
        )

    def forward(self, audio, mask):
        input_spec = self.mel_transforms.input(audio.squeeze(1))

        latent = self.generator.encoder(input_spec)
        quantize, _, vq_losses = self.vq(latent)
        vq_loss = vq_losses.mean()
        fake_audio = self.generator.decoder(quantize)

        stage = "train" if self.training else "val"
        self.log(
            f"{stage}/generator/vq",
            vq_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return fake_audio, vq_loss * 5
