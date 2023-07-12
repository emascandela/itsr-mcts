from typing import Dict, Any
import numpy as np
import random
from .figure_composer import GridComposer, SequenceGridComposer
from .image_processor import ImagenetteProcessor
from .generator import Generator
from .image_processor import distortions

TOY_FREE_SCENARIO_PARAMS = {
    "image_size": 32,
    "min_transformations": 1,
    "max_transformations": None,
}

TOY_CONSTRAINED_SCENARIO_PARAMS = {
    "image_size": 32,
    "min_transformations": 1,
    "max_transformations": None,
}

REAL_SCENARIO_PARAMS = {
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
}


def get_generator(scenario: str, split: str, seed: int = 42, generator_params: Dict[str, Any] = None) -> Generator:
    np.random.seed(seed)
    random.seed(seed)

    if scenario == "toy_free":
        generator_class = GridComposer
        generator_params = TOY_FREE_SCENARIO_PARAMS | generator_params
    elif scenario == "toy_constrained":
        generator_class = SequenceGridComposer
        generator_params = TOY_CONSTRAINED_SCENARIO_PARAMS | generator_params
    elif scenario == "real":
        generator_class = ImagenetteProcessor
        generator_params = REAL_SCENARIO_PARAMS | generator_params
    return generator_class(split=split, **generator_params)
