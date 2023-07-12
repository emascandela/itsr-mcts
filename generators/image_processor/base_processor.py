import glob
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Union

import numpy as np
import PIL.Image
import cv2

from ..generator import Generator
from ..image import ImagePair
from . import distortions
from .image import ProcessedImage


class ImageProcessor(Generator):
    def __init__(
        self,
        image_size: int = 224,
        transformations: List[Callable] = None,
        preprocessing_transformations: List[Callable] = [],
        min_transformations: int = 0,
        max_transformations: int = None,
        max_repetitions: int = 1,
    ):
        self.image_size = image_size
        self.min_transformations = min_transformations
        self.max_transformations = max_transformations
        self.max_repetitions = max_repetitions

        transformations = transformations or distortions.all_distortions
        self.transformations = transformations
        # self.transformations = [
        #     self.get_transformation_fn(fn) for fn in transformations
        # ]
        # print(self.transformations[0](np.random.uniform((100, 100, 3))))
        self.preprocessing_transformations = preprocessing_transformations

    @abstractmethod
    def get_random_image(self) -> np.ndarray:
        pass

    def init_image(self) -> ProcessedImage:
        image = self.get_random_image()
        image = cv2.resize(image, (self.image_size, self.image_size))
        return ProcessedImage(image)

    def apply_transformation(self, fn: Callable, image: ProcessedImage):
        # if fn not in self.allowed_transformations:
        #     raise Exception
        return image.apply(fn)

    def get_transformation_fn(self, fn: Callable) -> Callable:
        transformation_fn = partial(self.apply_transformation, fn)
        transformation_fn.__name__ = fn.__name__
        return transformation_fn

    def _allowed_transformations(
        self, image: ProcessedImage
    ) -> Union[List[Callable], np.ndarray]:
        if (
            self.max_transformations is not None
            and len(image.applied_transformations) >= self.max_transformations
        ):
            return []

        # allowed_transformations = list(
        #     np.repeat(self.transformations, self.max_repetitions)
        # )
        allowed_transformations = self.transformations * self.max_repetitions

        for tform in image.applied_transformations:
            if tform in allowed_transformations:
                allowed_transformations.remove(tform)
        return list(set(allowed_transformations))

    def get_random_pair(self) -> ImagePair:
        transformations = self.random_sequence(
            self.transformations,
            max_repetitions=self.max_repetitions,
            min_length=self.min_transformations,
            max_length=self.max_transformations,
        )

        return self.get_pair(
            source_transformations=[], target_transformations=transformations
        )

    @property
    def max_transformation_sequence_length(self) -> int:
        return self.max_transformations or self.max_repetitions * len(
            self.transformations
        )


class ImageDatasetProcessor(ImageProcessor):
    def __init__(self, *args, **kwargs):
        ImageProcessor.__init__(self, *args, **kwargs)

        self.image_paths = []

    def get_random_image(self) -> np.ndarray:
        path = np.random.choice(self.image_paths)
        image = PIL.Image.open(path)
        return np.asarray(image.convert("RGB"))


class ImagenetteProcessor(ImageDatasetProcessor):
    TRAIN_PATH: str = "data/imagenette2/*/n0[29,3]*/*.JPEG"
    VAL_PATH: str = "data/imagenette2/*/n0[1]*/*.JPEG"
    TEST_PATH: str = "data/imagenette2/*/n0[21]*/*.JPEG"

    def __init__(self, split: str = "train", *args, **kwargs):
        ImageDatasetProcessor.__init__(self, *args, **kwargs)

        if split == "train":
            path = self.TRAIN_PATH
        elif split == "val":
            path = self.VAL_PATH
        elif split == "test":
            path = self.TEST_PATH
        else:
            Exception(f"Unknown split {split}")

        self.image_paths = list(glob.glob(path))
