from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)

    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )

    if depth_first and include_root:
        fn(module=module, name=name)

    return module


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super(ResBlock, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels if idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs1.apply(self.init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs2.apply(self.init_weights)

    def forward(self, x):
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)

            if idx != 0 or self.in_channels == self.out_channels:
                x = xt + x
            else:
                x = xt

        return x

    def remove_parametrizations(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1)
            remove_parametrizations(c2)

    def init_weights(self, m):
        if type(m) == nn.Conv1d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.0)


class AdaIN(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        results = [block(x) for block in self.blocks]

        return torch.mean(torch.stack(results), dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block[1].remove_parametrizations()


class RefineGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        sampling_rate: int = 44100,
        hop_length: int = 256,
        downsample_rates: tuple[int] = (2, 2, 8, 8),
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
    ) -> None:
        super().__init__()

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.downsample_rates = downsample_rates
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope

        assert np.prod(downsample_rates) == np.prod(upsample_rates) == hop_length

        self.template_conv = weight_norm(
            nn.Conv1d(
                in_channels=1,
                out_channels=start_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        channels = start_channels

        self.downsample_blocks = nn.ModuleList([])
        for rate in downsample_rates:
            new_channels = channels * 2

            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=1 / rate, mode="linear"),
                    ResBlock(
                        in_channels=channels,
                        out_channels=new_channels,
                        kernel_size=7,
                        dilation=(1, 3, 5),
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                )
            )

            channels = new_channels

        self.mel_conv = weight_norm(
            nn.Conv1d(
                in_channels=num_mels,
                out_channels=channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        channels *= 2

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2

            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.output_conv = weight_norm(
            nn.Conv1d(
                in_channels=channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

    def remove_parametrizations(self) -> None:
        remove_parametrizations(self.template_conv)
        remove_parametrizations(self.mel_conv)
        remove_parametrizations(self.output_conv)

        for block in self.downsample_blocks:
            block[1].remove_parametrizations()

        for block in self.upsample_conv_blocks:
            block.remove_parametrizations()

    def forward(self, mel: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (torch.Tensor): [B, mel_bin, T]
            template (torch.Tensor): [B, 1, T]

        Returns:
            torch.Tensor: [B, 1, T]
        """

        x = self.template_conv(template)

        downs = []

        for block in self.downsample_blocks:
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            downs.append(x)
            x = block(x)

        x = torch.cat([x, self.mel_conv(mel)], dim=1)

        for upsample_block, conv_block, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            x = upsample_block(x)

            x = torch.cat([x, down], dim=1)
            x = conv_block(x)

        x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
        x = self.output_conv(x)
        x = torch.tanh(x)

        return x
