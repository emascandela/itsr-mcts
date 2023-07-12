import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import numpy as np
import PIL.Image


class Image(ABC):
    def __init__(self):
        self.applied_transformations = []

    @abstractmethod
    def copy(self) -> "Image":
        pass

    @abstractmethod
    def pil_image(self) -> PIL.Image.Image:
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    def apply(self, fn: Union[List[Callable], Callable]) -> "Image":
        if isinstance(fn, np.ndarray):
            fn = list(fn)
        elif not isinstance(fn, list):
            fn = [fn]

        for fn_i in fn:
            self.applied_transformations.append(fn_i)
            self.image = fn_i(self)
        return self

    @property
    @abstractmethod
    def applied_sequence_length(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: "Image"):
        pass


class ImagePair:
    def __init__(
        self,
        source_image: Image,
        target_image: Image,
        preprocessing_sequence: List[Callable] = [],
        gt_sequence: List[Callable] = [],
    ):
        self.source_image: Image = source_image
        self.target_image: Image = target_image

        self.preprocessing_sequence: List[Callable] = preprocessing_sequence
        self.transformation_sequence: List[Callable] = []
        self.gt_sequence = gt_sequence

    def copy(self) -> "ImagePair":
        other = ImagePair(
            source_image=self.source_image.copy(),
            target_image=self.target_image,
            preprocessing_sequence=self.preprocessing_sequence,
            gt_sequence=self.gt_sequence,
        )
        other.transformation_sequence = copy.copy(self.transformation_sequence)

        return other

    def apply_transformation(
        self, transformation_fn: Callable, inplace: bool = False
    ) -> "ImagePair":
        if not inplace:
            obj = self.copy()
        else:
            obj = self

        obj.transformation_sequence.append(transformation_fn)
        obj.source_image = obj.source_image.apply(transformation_fn)
        # obj.source_image = transformation_fn(obj.source_image)

        return obj

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.source_image.numpy(), self.target_image.numpy()

    def pil_image(
        self, single_image: bool = False
    ) -> Union[PIL.Image.Image, Tuple[PIL.Image.Image, PIL.Image.Image]]:
        source = self.source_image.pil_image()
        target = self.target_image.pil_image()

        if single_image:
            new_image = PIL.Image.new("RGB", (2 * source.size[0], source.size[1]))
            new_image.paste(source)
            new_image.paste(target, (source.size[0], 0))
            return new_image

        return source, target

    def issame(self) -> bool:
        return self.source_image == self.target_image
