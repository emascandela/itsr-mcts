import tensorflow as tf
from typing import List
from .backbone import Backbone


class TinyConv(Backbone):
    def __init__(self, blocks: List[int] = [32, 64]):
        self.blocks = blocks

    def build(self, input_shape):
        x = inp = tf.keras.layers.Input(input_shape)
        for b in self.blocks:
            x = tf.keras.layers.Conv2D(
                b, kernel_size=3, activation="relu", padding="same"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(
                b, kernel_size=3, activation="relu", padding="same"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D()(x)
        return tf.keras.models.Model(inp, x)

from . import backbone_factory
from torch import nn

class LeNet(nn.Module):
    def __init__(self, blocks: List[int] = [32, 64]):
        super().__init__()
        self.blocks = blocks

        layers = []
        inp_channels = 3
        for b in self.blocks:
            layers.append(nn.Conv2d(inp_channels, b, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(b))
            layers.append(nn.Conv2d(b, b, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(b))
            layers.append(nn.MaxPool2d(2))
            inp_channels = b

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

backbone_factory.register(LeNet, "lenet")