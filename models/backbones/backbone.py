from abc import ABC, abstractmethod
from typing import Dict


class Backbone:
    def __init__(self):
        pass

    @abstractmethod
    def build(self, input_shape):
        pass
