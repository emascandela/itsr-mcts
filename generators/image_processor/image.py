from typing import Callable, List, Union
import copy

import numpy as np
import PIL.Image

from ..image import Image


class ProcessedImage(Image):
    def __init__(self, image: np.ndarray):
        Image.__init__(self)

        self.image = image

    def copy(self) -> "ProcessedImage":
        other = ProcessedImage(np.copy(self.image))
        other.applied_transformations = copy.copy(self.applied_transformations)
        return other

    @property
    def applied_sequence_length(self):
        return len(self.applied_transformations)

    def apply(self, fn: Union[List[Callable], Callable]) -> "ProcessedImage":
        if isinstance(fn, np.ndarray):
            fn = list(fn)
        elif not isinstance(fn, list):
            fn = [fn]

        for fn_i in fn:
            self.applied_transformations.append(fn_i)
            self.image = fn_i(self.image)
        return self

    def pil_image(self) -> PIL.Image:
        return PIL.Image.fromarray(self.image)

    def numpy(self) -> np.ndarray:
        return self.image

    def __eq__(self, other: "ProcessedImage") -> bool:
        return np.all(other.image == self.image)
