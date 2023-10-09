import torch
from torch import nn
from transformers import HubertModel


class HubertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ll60k",
        freeze_backbone: bool = True,
        output_size: int = 1024,
    ):
        super().__init__()

        self.model = HubertModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.post = nn.Sequential(
            nn.Conv1d(
                self.model.config.hidden_size, output_size, kernel_size=3, padding=1
            ),
            nn.SiLU(),
            nn.Conv1d(output_size, output_size, stride=2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(output_size, output_size, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.ndim == 3:
            assert x.shape[1] == 1 and mask.shape[1] == 1
            x = x.squeeze(1)
            mask = mask.squeeze(1)

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.model(x, attention_mask=mask)
        else:
            x = self.model(x, attention_mask=mask)

        x = x.last_hidden_state.transpose(1, 2)
        x = self.post(x)

        return x


if __name__ == "__main__":
    model = HubertEncoder()
    x = torch.randn(1, 16000)
    y = model(x)
    print(y.shape)
