from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from vocos.heads import FourierHead
from vocos.models import AdaLayerNorm, Backbone
from vocos.spectral_ops import ISTFT


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        dilation: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
            dilation=dilation,
        )  # depthwise conv

        self.adanorm = adanorm_num_embeddings is not None

        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_embedding_id: Optional[torch.Tensor] = None,
        apply_residual: bool = True,
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        if apply_residual:
            x = residual + x

        return x


class ParallelConvNeXtBlock(nn.Module):
    def __init__(self, kernel_sizes: list[int], *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(kernel_size=kernel_size, *args, **kwargs)
                for kernel_size in kernel_sizes
            ]
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.stack(
            [block(x, cond_embedding_id, apply_residual=False) for block in self.blocks]
            + [x],
            dim=1,
        ).mean(dim=1)


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
        dilation_cycle (int, optional): Dilation cycle for ConvNeXtBlock. Defaults to 4.
        kernel_sizes (tuple, optional): Kernel sizes for ParallelConvNeXtBlock. Defaults to (7,).
    """  # noqa: E501

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        dilation_cycle=4,
        kernel_sizes=(7,),
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None

        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        layer_scale_init_value = layer_scale_init_value or 1 / num_layers

        block_fn = (
            partial(ConvNeXtBlock, kernel_size=kernel_sizes[0])
            if len(kernel_sizes) == 1
            else partial(ParallelConvNeXtBlock, kernel_sizes=kernel_sizes)
        )

        self.convnext = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """  # noqa: E501

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """  # noqa: E501

        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        phase = torch.atan2(y, x)
        S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        # S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class VocosGenerator(nn.Module):
    def __init__(self, backbone: Backbone, head: FourierHead):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x[:, None, :]
