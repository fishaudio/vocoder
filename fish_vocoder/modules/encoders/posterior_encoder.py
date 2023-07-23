"""
Borrowed from RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
"""

import torch
from torch import nn

from fish_vocoder.utils.mask import sequence_mask


class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        dilation_cycle,
        n_layers,
        p_dropout=0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** (i % dilation_cycle)
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            # Gate
            in_act = x_in
            t_act = torch.tanh(in_act[:, : n_channels_tensor[0], :])
            s_act = torch.sigmoid(in_act[:, n_channels_tensor[0] :, :])
            acts = t_act * s_act

            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size=5,
        dilation_rate=1,
        dilation_cycle=1,
        n_layers=16,
        mode="vqvae",
    ):
        super().__init__()

        assert mode in ["vae", "bnvae", "vqvae"]

        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dilation_cycle = dilation_cycle
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dilation_cycle=dilation_cycle,
            n_layers=n_layers,
        )
        self.proj = nn.Conv1d(
            hidden_channels, out_channels * 2 if mode != "vqvae" else out_channels, 1
        )

        if mode == "bnvae":
            self.mu_bn = nn.BatchNorm1d(out_channels, affine=True)
            self.mu_bn.weight.requires_grad = False
            self.mu_bn.weight.fill_(0.5)

    def forward(self, x, x_lengths=None):
        if x_lengths is not None:
            x_mask = sequence_mask(x_lengths, x.size(2))[:, None].to(
                device=x.device, dtype=x.dtype
            )
        else:
            x_mask = torch.ones(x.size(0), 1, x.size(2), dtype=x.dtype, device=x.device)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        x = self.proj(x) * x_mask

        if self.mode in ["bnvae", "vae"]:
            mean, logvar = torch.split(x, self.out_channels, dim=1)
            logvar = torch.clamp(logvar, min=-30, max=20)

            if self.mode == "bnvae":
                mean = self.mu_bn(mean)

            if self.training:
                z = (mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)) * x_mask
            else:
                z = mean * x_mask

            return z, mean, logvar, x_mask

        elif self.mode == "vqvae":
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
