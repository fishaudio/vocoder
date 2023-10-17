import torch
from encodec.quantization.core_vq import ResidualVectorQuantization, VectorQuantization

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
    def __init__(
        self,
        latent_size: int,
        codebook_size: int,
        num_quantizers: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.latent_size = latent_size
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        if num_quantizers > 1:
            self.vq = ResidualVectorQuantization(
                dim=latent_size,
                codebook_size=codebook_size,
                num_quantizers=num_quantizers,
                kmeans_init=False,
            )
        else:
            self.vq = VectorQuantization(
                dim=latent_size,
                codebook_size=codebook_size,
                kmeans_init=False,
            )

    def forward(self, audio, mask, input_spec=None):
        latent = self.generator.encoder(audio, mask)
        quantize, _, vq_loss = self.vq(latent)

        if self.num_quantizers > 1:
            vq_loss = vq_loss.mean()

        fake_audio = self.generator.decoder(quantize)

        assert abs(fake_audio.size(2) - audio.size(2)) <= self.hop_length

        if fake_audio.size(2) > audio.size(2):
            fake_audio = fake_audio[:, :, : audio.size(2)]
        else:
            fake_audio = torch.nn.functional.pad(
                fake_audio, (0, audio.size(2) - fake_audio.size(2))
            )

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

        return fake_audio, 0  # vq_loss * 5
