from torch import nn
import timm
from . import backbone_factory


class EfficientNetV2B0(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = m = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=0, global_pool='')

    def forward(self, x):
        return self.features(x)


backbone_factory.register(EfficientNetV2B0, "efn-b0")
