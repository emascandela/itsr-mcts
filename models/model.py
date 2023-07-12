from typing import Type, Dict

# import tensorflow as tf
import datetime
import os

import torch

# from torch import nn
from models.archs.arch import Arch
from utils import get_logger


from generators.generator import Generator
from models.preprocessors.preprocessor import Preprocessor
from typing import Dict, Any, Type

from .backbones import backbone_factory
from .archs import arch_factory


class Model:
    def __init__(
        self,
        *,
        name: str,
        step: int,
        arch: str,
        backbone: str,
        generator_class: Type[Generator],
        generator_params: Dict[str, Any] = {},
        train_generator_params: Dict[str, Any] = {},
        val_generator_params: Dict[str, Any] = {},
        test_generator_params: Dict[str, Any] = {},
        preprocessor_class: Type[Preprocessor],
        preprocessor_params: Dict[str, Any] = {},
        train_config: Dict[str, Any] = {},
        evaluate_config: Dict[str, Any] = {},
    ):
        self.name = name
        self.step = step
        self.logger = get_logger(self.get_log_path())
        self.generator_class = generator_class
        self.generator_params = generator_params
        self.train_generator_params = {
            "split": "train",
            **generator_params,
            **train_generator_params,
        }
        self.val_generator_params = {
            "split": "val",
            **generator_params,
            **val_generator_params,
        }
        self.test_generator_params = {
            "split": "test",
            **generator_params,
            **test_generator_params,
        }

        self.preprocessor_class = preprocessor_class
        self.preprocessor_params = preprocessor_params
        self.preprocessor = self.preprocessor_class(**self.preprocessor_params)
        self.train_config = train_config
        self.evaluate_config = evaluate_config

        self.arch = arch
        self.backbone = backbone

    def build_model(self):
        backbone = backbone_factory.get(self.backbone)
        shape = (3, self.val_generator.image_size, self.val_generator.image_size)
        model = arch_factory.get(
            self.arch,
            backbone=backbone,
            input_shape=shape,
            n_classes=len(self.val_generator.transformations),
        )
        model = model.cuda()
        return model

    def get_model_path(self, name: str = "model", weights_dir: str = "experiments"):
        path = os.path.join(weights_dir, self.name, name + (self.step or ""))
        return path

    def get_log_path(self, name: str = None, weights_dir: str = "experiments"):
        name = name or self.name
        path = os.path.join(weights_dir, name, "output" + (self.step or "") + ".log")
        return path

    def get_tensorboard_path(self, name: str = None, weights_dir: str = "experiments"):
        name = name or self.name
        path = os.path.join(
            weights_dir, name, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        return path

    def save_weights(self, name: str = "model", weights_dir: str = "experiments"):
        path = self.get_model_path(name=name, weights_dir=weights_dir)

        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        torch.save(self.model.state_dict(), path)

    def load_weights(self, name: str = "model", weights_dir: str = "experiments"):
        path = self.get_model_path(name=name, weights_dir=weights_dir)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
