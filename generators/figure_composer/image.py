import copy
from typing import Tuple, Type, Callable

import numpy as np
import PIL.Image

from ..image import Image
from .figures import Figure


class ComposedImage(Image):
    def __init__(
        self,
        size: int = 224,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        Image.__init__(self)

        self.size = size
        self.color = color
        self.figures = []
        self.removed_figures = []
        self.added_figures = []

    @property
    def applied_sequence_length(self):
        return len(self.removed_figures) + len(self.added_figures)

    def copy(self) -> "ComposedImage":
        new_image = ComposedImage(size=self.size, color=self.color)
        new_image.figures = [*self.figures]
        new_image.removed_figures = [*self.removed_figures]
        new_image.added_figures = [*self.added_figures]
        new_image.applied_transformations = [*self.applied_transformations]
        return new_image

    def add_figure(self, figure: Figure) -> "ComposedImage":
        self.figures.append(figure)
        self.added_figures.append(figure)
        return self

    def remove_figure(self, figure: Figure) -> "ComposedImage":
        self.figures.remove(figure)
        self.removed_figures.append(figure)
        return self

    def pil_image(self) -> PIL.Image.Image:
        image = PIL.Image.new("RGBA", (self.size, self.size), color=self.color + (255,))

        for fig in self.figures:
            image = fig.draw(image)

        return image

    def numpy(self) -> np.ndarray:
        image = self.pil_image()
        return np.asarray(image.convert("RGB"))

    def remove_first_figure(self, figure_class: Type[Figure]) -> "ComposedImage":
        for fig in self.figures:
            if isinstance(fig, figure_class):
                self.figures.remove(fig)
                break
        return self

    def __eq__(self, other: "ComposedImage"):
        if not isinstance(other, self.__class__):
            return False

        figures = sorted(self.figures, key=lambda x: x.location[0] * x.location[1])
        other_figures = sorted(
            other.figures, key=lambda x: x.location[0] * x.location[1]
        )

        if len(figures) != len(other_figures):
            return False

        for fig, other_fig in zip(figures, other_figures):
            if fig != other_fig:
                return False

        return True


class ComposedGridImage(ComposedImage):
    def __init__(self, grid_size: int, **kwargs):
        ComposedImage.__init__(self, **kwargs)

        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), None, dtype=object)

    def copy(self) -> "ComposedGridImage":
        new_image = ComposedGridImage(
            size=self.size, color=self.color, grid_size=self.grid_size
        )
        new_image.figures = [*self.figures]
        new_image.removed_figures = [*self.removed_figures]
        new_image.added_figures = [*self.added_figures]
        # new_image.actions = copy.deepcopy(self.added_figures)
        new_image.grid = np.copy(self.grid)
        return new_image

    def add_figure(self, figure: Figure, position: Tuple[int, int]):
        if position is None:
            empty_grid = self.grid == None
            if not np.any(empty_grid):
                return self

            y = np.nonzero(np.any(empty_grid, axis=1))[0][0]
            x = np.nonzero(empty_grid[y])[0][0]
            position = (y, x)

        if self.grid[position] is not None:
            return self

        location = (
            int((position[1] + 0.5) / (self.grid_size) * self.size),
            int((position[0] + 0.5) / (self.grid_size) * self.size),
        )

        self.grid[position] = figure
        super().add_figure(figure)

        figure.location = location
        return self

    def remove_figure(self, position: Tuple[int, int]):
        if position is None:
            empty_grid = self.grid == None
            if np.all(empty_grid):
                return self

            y = np.nonzero(np.any(~empty_grid, axis=1))[0][-1]
            x = np.nonzero(~empty_grid[y])[0][-1]
            position = (y, x)

        figure = self.grid[position]
        if figure is not None:
            super().remove_figure(figure)
            self.grid[position] = None

        return self
