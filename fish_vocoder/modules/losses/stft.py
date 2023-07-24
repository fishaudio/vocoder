# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
from torch import nn


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    spec = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        return_complex=True,
        pad_mode="reflect",
    )
    spec = torch.view_as_real(spec)

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(spec.pow(2).sum(-1), min=1e-6)).transpose(2, 1)


class SpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initialize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """  # noqa: E501

        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """  # noqa: E501

        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()

        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", window(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """

        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, resolutions, window=torch.hann_window):
        super(MultiResolutionSTFTLoss, self).__init__()

        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
