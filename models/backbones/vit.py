from . import backbone_factory
import torch
from torch import nn
from torchvision.models.vision_transformer import _vision_transformer



class ViTB(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            image_size=112,
            weights=None,
            progress=False
        )

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x


backbone_factory.register(ViTB, "vit-b")
