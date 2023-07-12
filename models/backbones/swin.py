from . import backbone_factory
from torch import nn


from torchvision.models.swin_transformer import swin_v2_t


class SwinT(nn.Module):
    def __init__(self):
        super().__init__()
        model = swin_v2_t()
        self.features = model.features

    def forward(self, x):
        return self.features(x)


backbone_factory.register(SwinT, "swin-t")
