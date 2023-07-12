from . import arch_factory
from typing import Tuple, List, Type

# import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from ..backbones.backbone import Backbone


class SiameseFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        backbone: Backbone,
        fusion_method: str = "sub",
    ):
        nn.Module.__init__(self)
        self.input_shape = input_shape
        self.fusion_method = fusion_method

        self.backbone = backbone
        out_shape = self.backbone(torch.randn(1, *input_shape)).data.shape
        self.out_channels = (out_shape[1] * 2) if len(out_shape) == 3 else out_shape[1]

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor):
        x_o = self.backbone(source_image)
        x_t = self.backbone(target_image)
        x = x_o - x_t

        if len(x.shape) == 4:
            x = torch.cat(
                [
                    self.global_avg_pooling(x),
                    self.global_max_pooling(x),
                ],
                dim=1,
            )
        x = self.flatten(x)
        return x


class FCN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        backbone: nn.Module,
        fusion_method: str = "sub",
        head_units: List[int] = [128],
    ):
        nn.Module.__init__(self)
        self.feature_extractor = SiameseFeatureExtractor(
            input_shape=input_shape, backbone=backbone, fusion_method=fusion_method
        )
        self.n_classes = n_classes
        self.head_units = head_units

        self.head = nn.Sequential(
            *[
                self.get_block(inp, out)
                for inp, out in zip(
                    [self.feature_extractor.out_channels] + self.head_units,
                    self.head_units,
                )
            ],
            nn.Linear(self.head_units[-1], self.n_classes)
        )

    def get_block(self, inp_features, out_features):
        return nn.Sequential(
            *[
                nn.Linear(inp_features, out_features),
                nn.GELU(),
                nn.BatchNorm1d(out_features),
                nn.Dropout(0.3),
            ]
        )

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor):
        x = self.feature_extractor(source_image, target_image)
        x = self.head(x)
        return x


arch_factory.register(FCN, "fcn")


class FCNActorCritic(FCN):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        backbone: Backbone,
        fusion_method: str = "sub",
        neck_units: List[int] = [128],
        actor_head_units: List[int] = [128],
        value_head_units: List[int] = [128],
    ):
        nn.Module.__init__(self)
        self.feature_extractor = SiameseFeatureExtractor(
            input_shape=input_shape, backbone=backbone, fusion_method=fusion_method
        )
        self.n_classes = n_classes
        self.actor_head_units = actor_head_units
        self.value_head_units = value_head_units
        self.neck_units = neck_units

        self.neck = nn.Sequential(
            *[
                self.get_block(inp, out)
                for inp, out in zip(
                    [self.feature_extractor.out_channels * 2] + self.neck_units,
                    self.neck_units,
                )
            ]
        )

        self.actor_head = nn.Sequential(
            *[
                self.get_block(inp, out)
                for inp, out in zip(
                    [self.neck_units[-1]] + self.actor_head_units,
                    self.actor_head_units,
                )
            ],
            nn.Linear(self.actor_head_units[-1], self.n_classes)
        )

        self.value_head = nn.Sequential(
            *[
                self.get_block(inp, out)
                for inp, out in zip(
                    [self.neck_units[-1]] + self.value_head_units,
                    self.value_head_units,
                )
            ],
            nn.Linear(self.value_head_units[-1], 1)
        )

        self.softmax = torch.nn.Softmax(dim=1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(
                module.weight,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(
                module.weight,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        training: bool = True,
    ):
        x = self.feature_extractor(source_image, target_image)
        x = self.neck(x)
        action_probs = self.actor_head(x)
        value = self.value_head(x)

        if not training:
            action_probs = self.softmax(action_probs)
        return action_probs, value


arch_factory.register(FCNActorCritic, "actor-critic")
