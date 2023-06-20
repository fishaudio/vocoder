from torch import Tensor, nn
from vocos.heads import FourierHead
from vocos.models import Backbone


class VocosGenerator(nn.Module):
    def __init__(self, backbone: Backbone, head: FourierHead):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, template=None) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x[:, None, :]
