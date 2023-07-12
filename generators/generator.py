import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np

from .image import Image, ImagePair


class Generator(ABC):
    def __init__(
        self, min_transformations: int = None, max_transformations: int = None
    ):
        self.min_transformations = min_transformations
        self.max_transformations = max_transformations
        self.transformations = []

    @staticmethod
    def apply(image: Image, transformations: List[Callable]) -> Image:
        for tform in transformations:
            image = tform(image)
        return image

    @staticmethod
    def random_sequence(
        x,
        max_repetitions: int = 1,
        max_length: int = None,
        min_length: int = 0,
    ):
        x = copy.copy(x)
        if max_repetitions > 0:
            x = np.repeat(x, max_repetitions)

        max_length = max_length or len(x)
        np.random.shuffle(x)
        n = np.random.randint(min_length, max_length + 1)
        x = np.random.choice(x, n, replace=max_repetitions == -1)
        return x

    def allowed_transformations(
        self, image: Image, return_mask: bool = False
    ) -> Union[List[Callable], np.ndarray]:
        allowed_fns = self._allowed_transformations(image)

        if return_mask:
            allowed_mask = np.array([t in allowed_fns for t in self.transformations])
            return allowed_fns, allowed_mask

        return allowed_fns

    def get_pair(
        self,
        source_transformations: List[Callable],
        target_transformations: List[Callable],
    ):
        source_image = self.init_image()
        source_image = source_image.apply(source_transformations)
        source_image.applied_transformations = []
        target_image = source_image.copy()
        target_image = target_image.apply(target_transformations)
        # target_image.applied_transformations = []

        pair = ImagePair(
            source_image,
            target_image,
            preprocessing_sequence=source_transformations,
            gt_sequence=target_transformations,
        )
        return pair

    @abstractmethod
    def get_random_pair(self) -> ImagePair:
        pass

    @property
    @abstractmethod
    def max_transformation_sequence_length(self) -> int:
        pass
