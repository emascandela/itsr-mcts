from . import backbone_factory
from torch import nn


from torchvision.models import convnext_tiny


class ConvNeXtT(nn.Module):
    def __init__(self):
        super().__init__()
        model = convnext_tiny()
        self.features = model.features

    def forward(self, x):
        return self.features(x)


backbone_factory.register(ConvNeXtT, "convnext-t")
