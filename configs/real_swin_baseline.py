from typing import Sequence
from generators.figure_composer.figures import Triangle, Square, Circle
from models import backbones
from models import FCNModel
from generators import ImagenetteProcessor
from models.preprocessors import FCNPreprocessor
from generators.image_processor import distortions

conf = dict(
    model_class=FCNModel,
    arch="fcn",
    backbone="swin-t",
    generator_class=ImagenetteProcessor,
    generator_params={
        "image_size": 112,
        "min_transformations": 1,
        "max_repetitions": 2,
    },
    preprocessor_class=FCNPreprocessor,
    preprocessor_params={},
    train_config={
        "epochs": 40,
        "batch_size": 16,
        "steps_per_epoch": 5000,
        "validation_steps": 500,
        "evaluation_steps": 1000,
        "optimizer_params": {
            "lr": 1e-3,
        },
        "rollout_threads": 2048,
        "rollout_batch_size": 256,
    },
)
