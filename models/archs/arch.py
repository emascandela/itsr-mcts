from abc import ABC, abstractmethod
from typing import Tuple


class Arch(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build(input_shape: Tuple[int, int, int], n_classes: int):
        pass
