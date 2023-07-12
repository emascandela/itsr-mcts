from typing import Sequence
from generators.figure_composer.figures import Triangle, Square, Circle
from models.archs import FCN
from models import backbones
from models import FCNModel
from generators import GridComposer, SequenceGridComposer
from models.preprocessors import FCNPreprocessor
from generators.image_processor import distortions

conf = dict(
    model_class=FCNModel,
    arch="fcn",
    backbone="lenet",
    generator_class=SequenceGridComposer,
    generator_params={
        "image_size": 32,
        "min_transformations": 1,
    },
    preprocessor_class=FCNPreprocessor,
    preprocessor_params={},
    train_config={
        "epochs": 25,
        "batch_size": 128,
        "steps_per_epoch": 5000,
        "validation_steps": 500,
        "evaluation_steps": 1000,
        "optimizer_params": {
            "lr": 1e-3,
        },
        "rollout_threads": 2048,
        "rollout_batch_size": 1024,
    },
)
