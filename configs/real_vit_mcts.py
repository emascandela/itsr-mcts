from models import MCTSModel
from generators import ImagenetteProcessor
from models.preprocessors import MCTSPreprocessor
from generators.image_processor import distortions

conf = dict(
    model_class=MCTSModel,
    arch="actor-critic",
    backbone="vit-b",
    generator_class=ImagenetteProcessor,
    generator_params={
        "image_size": 112,
        "min_transformations": 1,
        "max_transformations": None,
        "max_repetitions": 2,
        "transformations": [
            distortions.blur,
            distortions.invert,
            distortions.erode,
            distortions.dilate,
            distortions.rotate90,
        ],
    },
    preprocessor_class=MCTSPreprocessor,
    preprocessor_params={},
    train_config={
        "general": {
            "replay_size": 10000,
            "batch_size": 16,
            "evaluation_threads": 512,
            "evaluation_batch_size": 128,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 5.0,
            "c_puct": 1.0,
            "num_simulations": 200,
            "rollout_threads": 128,
            "rollout_batch_size": 64,
            "steps_per_epoch": 10,
            "num_episodes": 1000,
            "num_simulations": 100,
            "temperature": 1.0,
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
                "epochs": 10,
                "generator_params": {
                    "max_transformations": 7,
                },
            },
            {
                "epochs": 10,
                "generator_params": {
                    "max_transformations": 10,
                },
            },
        ],
    },
)
