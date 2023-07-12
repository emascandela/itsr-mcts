from typing import Sequence
from generators.figure_composer.figures import Triangle, Square, Circle
from models import backbones
from models import MCTSModel
from generators import GridComposer, SequenceGridComposer
from models.preprocessors import MCTSPreprocessor
from generators.image_processor import distortions

conf = dict(
    model_class=MCTSModel,
    arch="actor-critic",
    backbone="lenet",
    generator_class=SequenceGridComposer,
    generator_params={
        "image_size": 32,
        "min_transformations": 1,
    },
    preprocessor_class=MCTSPreprocessor,
    preprocessor_params={},
    train_config={
        "general": {
            "batch_size": 128,
            "rollout_threads": 2048,
            "rollout_batch_size": 1024,
            "replay_size": 10000,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 5.0,
            "c_puct": 1.0,
            "steps_per_epoch": 10,
            "temperature": 1.0,
            "num_simulations": 100,
            "num_episodes": 1000,
            "optimizer_params": {
                "lr": 1e-3,
                "momentum": 0.8,
            },
        },
        "curriculum": [
            {
                "epochs": 5,
                "generator_params": {
                    "max_transformations": 4,
                },
            },
            {
                "epochs": 30,
                "generator_params": {
                    "max_transformations": 8,
                },
            },
        ],
    },
)
