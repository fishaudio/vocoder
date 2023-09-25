from typing import Optional

from torch import Tensor, nn


class Pad(nn.Module):
    def __init__(
        self,
        multiple_of: Optional[int] = None,
        target_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert (
            multiple_of is not None or target_length is not None
        ), "Either multiple_of or target_length must be specified."
        assert (
            multiple_of is None or target_length is None
        ), "Only one of multiple_of or target_length must be specified."

        self.multiple_of = multiple_of
        self.target_length = target_length

    def forward(self, waveform: Tensor) -> Tensor:
        if self.multiple_of is not None:
            pad = self.multiple_of - (waveform.shape[-1] % self.multiple_of)

            if pad == self.multiple_of:
                return waveform
        else:
            pad = self.target_length - waveform.shape[-1]

        return nn.functional.pad(waveform, (pad // 2, pad - (pad // 2)), "constant")
