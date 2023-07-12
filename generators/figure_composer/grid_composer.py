from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import math
import numpy as np

from ..generator import Generator
from ..image import ImagePair
from .figures import Circle, Figure, Square, Triangle
from .image import ComposedGridImage


class GridComposer(Generator):
    def __init__(
        self,
        grid_size: int = 3,
        min_transformations: int = None,
        max_transformations: int = None,
        figures: Figure = [Triangle, Square, Circle],
        colors: str = ["red", "green", "blue"],
        image_size: int = 224,
        figure_size: float = 0.8,
        allow_removal_after_insertion: bool = False,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        split: str = "test",
    ):
        self.image_size = image_size
        self.background_color = background_color

        self.figure_classes = figures
        self.colors = colors
        self.grid_size = grid_size
        self.min_transformations = min_transformations
        self.max_transformations = max_transformations
        self.split = split
        self.allow_removal_after_insertion = allow_removal_after_insertion

        self.figure_size = int(self.image_size / self.grid_size * figure_size)

        self.insert_functions = self.get_insert_functions()
        self.remove_functions = self.get_remove_functions()
        self.transformations = self.insert_functions + self.remove_functions

    @property
    def max_transformation_sequence_length(self) -> int:
        return self.max_transformations or (2 * self.grid_size**2)

    def init_image(self) -> ComposedGridImage:
        return ComposedGridImage(
            size=self.image_size, color=self.background_color, grid_size=self.grid_size
        )

    def _allowed_transformations(
        self,
        image: ComposedGridImage,
        return_inserts_removes: bool = False,
        ignore_limits: bool = False,
    ) -> Union[List[Callable], np.ndarray]:
        allowed_inserts = []
        allowed_removes = []

        if not (
            not ignore_limits
            and self.max_transformations is not None
            and image.applied_sequence_length >= self.max_transformations
        ):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if image.grid[i, j] is None:
                        for c in self.colors:
                            for fig_class in self.figure_classes:
                                allowed_inserts.append(
                                    self._get_insert_function(
                                        figure_class=fig_class,
                                        color=c,
                                        grid_position=(i, j),
                                    )
                                )
                    else:
                        if (
                            len(image.added_figures) == 0
                            or self.allow_removal_after_insertion
                        ):
                            # Remove this in case want to delete figures once
                            allowed_removes.append(self._get_remove_function((i, j)))
        if return_inserts_removes:
            return allowed_inserts + allowed_removes, allowed_inserts, allowed_removes
        return allowed_inserts + allowed_removes

    def _get_insert_function(
        self,
        figure_class: Type[Figure],
        color: str,
        grid_position: Tuple[int, int] = None,
    ):
        fn_name = f"add_{color}_{figure_class.__name__.lower()}"
        if grid_position is not None:
            fn_name += f"_{grid_position[0]}_{grid_position[1]}"

        fn = getattr(self, fn_name, None)
        if fn is None:
            fn = partial(
                self.add_figure,
                position=grid_position,
                figure_class=figure_class,
                color=color,
            )
            fn.__name__ = fn_name
            setattr(self, fn_name, fn)
        return fn

    def _get_remove_function(self, grid_position: Tuple[int, int] = None):
        fn_name = f"remove"
        if grid_position is not None:
            fn_name += f"_{grid_position[0]}_{grid_position[1]}"
        fn = getattr(self, fn_name, None)
        if fn is None:
            fn = partial(
                self.remove_figure,
                position=grid_position,
            )
            fn.__name__ = fn_name
            setattr(self, fn_name, fn)
        return fn

    def get_insert_functions(self) -> List[Callable]:
        insert_functions = []
        for f in self.figure_classes:
            for c in self.colors:
                for gs_i in range(self.grid_size):
                    for gs_j in range(self.grid_size):
                        fn = self._get_insert_function(f, c, (gs_i, gs_j))
                        insert_functions.append(fn)
        return insert_functions

    def get_remove_functions(self) -> List[Callable]:
        remove_functions = []
        for gs_i in range(self.grid_size):
            for gs_j in range(self.grid_size):
                fn = self._get_remove_function(grid_position=(gs_i, gs_j))
                remove_functions.append(fn)
        return remove_functions

    def remove_figure(
        self,
        image: ComposedGridImage,
        position: Optional[Tuple[int, int]] = None,
    ):
        image.remove_figure(position)
        return image

    def add_random_figure(
        self,
        image: ComposedGridImage,
        position: Optional[Tuple[int, int]] = None,
    ):
        figure_class = np.random.choice(self.figure_classes)
        color = np.random.choice(self.colors)

        return self.add_figure(
            image=image,
            figure_class=figure_class,
            color=color,
            position=position,
        )

    def add_figure(
        self,
        image: ComposedGridImage,
        figure_class: Figure,
        color: Tuple[int, int, int] = (255, 255, 255),
        position: Optional[Tuple[int, int]] = None,
    ):

        figure = figure_class(
            location=None, color=color, size=self.figure_size, rotation=0
        )
        image.add_figure(figure, position)

        return image

    def get_random_pair(self) -> ImagePair:
        sequence_length = np.random.randint(
            self.min_transformations, self.max_transformation_sequence_length + 1
        )

        min_initial_figures = max(0, sequence_length - self.grid_size**2)
        n_initial_figures = np.random.randint(
            min_initial_figures, self.grid_size**2 + 1
        )

        min_removes = max(
            0,
            math.ceil(
                (sequence_length - (self.grid_size**2 - n_initial_figures)) / 2
            ),
        )
        max_removes = min(sequence_length, n_initial_figures)
        n_removes = np.random.randint(min_removes, max_removes + 1)
        n_inserts = sequence_length - n_removes

        # print("")
        # print("N Initial", n_initial_figures)
        # print("Min removes", min_removes)
        # print("Max removes", max_removes)
        # print("N Removes", n_removes)
        # print("N Inserts", n_inserts)

        target_image = self.init_image()

        for _ in range(n_initial_figures):
            _, allowed_inserts, _ = self._allowed_transformations(
                target_image, return_inserts_removes=True, ignore_limits=True
            )
            tf = np.random.choice(allowed_inserts)
            target_image = target_image.apply(tf)
        target_image.added_figures = []
        preprocessing_sequence = target_image.applied_transformations
        target_image.applied_transformations = []
        source_image = target_image.copy()

        for _ in range(n_removes):
            _, _, allowed_removes = self._allowed_transformations(
                target_image, return_inserts_removes=True
            )
            tf = np.random.choice(allowed_removes)
            target_image = target_image.apply(tf)

        for _ in range(n_inserts):
            _, allowed_inserts, _ = self._allowed_transformations(
                target_image, return_inserts_removes=True
            )
            tf = np.random.choice(allowed_inserts)
            target_image = target_image.apply(tf)

        pair = ImagePair(
            source_image,
            target_image,
            preprocessing_sequence=preprocessing_sequence,
            gt_sequence=target_image.applied_transformations,
        )

        if pair.issame():
            # Ensures that the generated pair is not the same
            return self.get_random_pair()

        return pair

        # Creates the pre image adding figures in random positions of the grid
        pre_mask = np.zeros(self.grid_size**2, dtype=bool)
        pre_indices = np.random.choice(
            self.grid_size**2, size=np.random.randint(self.grid_size**2 + 1)
        )
        pre_mask[pre_indices] = True
        pre_mask = pre_mask.reshape((self.grid_size, self.grid_size))

        target_image = self.init_image()
        for position in zip(*np.where(pre_mask)):
            target_image = self.add_random_figure(target_image, position=position)

        target_image.added_figures = []
        target_image.removed_figures = []
        source_image = target_image.copy()

        # Removes some of the figures randomly
        if len(pre_indices) > 0:
            remove_mask = np.zeros(self.grid_size**2, dtype=bool)
            if self.max_transformations is not None:
                max_removes = min(self.max_transformations, len(pre_indices))
            else:
                max_removes = len(pre_indices)
            remove_indices = np.random.choice(
                pre_indices,
                np.random.randint(
                    (1 if len(pre_indices) == self.grid_size**2 else 0),
                    max_removes + 1,
                ),
            )

            remove_mask[remove_indices] = True
            remove_mask = remove_mask.reshape((self.grid_size, self.grid_size))

            for position in zip(*np.where(remove_mask)):
                target_image = self.remove_figure(target_image, position=position)
                # target_image.applied_transformations.append("remove")
        else:
            remove_indices = []

        # Inserts figures in random positions
        full_indices = set(pre_indices) - set(remove_indices)
        empty_indices = list(set(range(self.grid_size**2)) - full_indices)

        if len(empty_indices) > 0:
            if self.max_transformations is not None:
                max_inserts = min(
                    self.max_transformations - len(remove_indices), len(empty_indices)
                )
            else:
                max_inserts = len(empty_indices)
            post_indices = np.random.choice(
                empty_indices,
                np.random.randint(
                    (1 if len(remove_indices) == 0 else 0), max_inserts + 1
                ),
            )
            post_mask = np.zeros(self.grid_size**2, dtype=bool)
            post_mask[post_indices] = True
            post_mask = post_mask.reshape((self.grid_size, self.grid_size))

            for position in zip(*np.where(post_mask)):
                target_image = self.add_random_figure(target_image, position=position)
                # target_image.applied_transformations.append("insert")

        pair = ImagePair(source_image, target_image)

        return pair
