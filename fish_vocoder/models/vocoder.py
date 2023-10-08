import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torchaudio.functional import resample
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

from fish_vocoder.data.transforms.spectrogram import LogMelSpectrogram
from fish_vocoder.utils.viz import plot_mel


class VocoderModel(L.LightningModule):
    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_mels: int,
    ):
        super().__init__()

        # Base parameters
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_mels = num_mels

        # Mel-Spectrogram for visualization
        self.viz_mel_transform = LogMelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.num_mels,
        )

    @torch.no_grad()
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def pesq(self, y_hat, y, sr=16000):
        y_hat = resample(y_hat, self.sampling_rate, sr)
        y = resample(y, self.sampling_rate, sr)

        return perceptual_evaluation_speech_quality(y_hat, y, sr, "wb").mean()

    @torch.no_grad()
    def report_val_metrics(self, y_g_hat, y, lengths):
        # PESQ
        pesq = self.pesq(y_g_hat, y)

        self.log(
            "val/metrics/pesq",
            pesq,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Mel-Spectrogram
        y_mel = self.viz_mel_transform(y.squeeze(1))
        y_g_hat_mel = self.viz_mel_transform(y_g_hat.squeeze(1))

        for idx, (mel, gen_mel, audio, gen_audio, audio_len) in enumerate(
            zip(y_mel, y_g_hat_mel, y.detach().cpu(), y_g_hat.detach().cpu(), lengths)
        ):
            mel_len = audio_len // self.hop_length

            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        "wavs": [
                            wandb.Audio(
                                audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="gt",
                            ),
                            wandb.Audio(
                                gen_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="prediction",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)
