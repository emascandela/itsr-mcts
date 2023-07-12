from typing import Callable, List, Union

import numpy as np

from ..image import ImagePair
from .grid_composer import GridComposer
from .image import ComposedGridImage


class SequenceGridComposer(GridComposer):
    def get_insert_functions(self) -> List[Callable]:
        insert_functions = []
        for f in self.figure_classes:
            for c in self.colors:
                fn = self._get_insert_function(figure_class=f, color=c)
                insert_functions.append(fn)

        return insert_functions

    def get_remove_functions(self) -> List[Callable]:
        remove_functions = [self._get_remove_function()]
        return remove_functions

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
            and image.applied_sequence_length > self.max_transformations
        ):
            if len(image.figures) < self.grid_size**2:
                for c in self.colors:
                    for fig_class in self.figure_classes:
                        allowed_inserts.append(
                            self._get_insert_function(
                                figure_class=fig_class,
                                color=c,
                            )
                        )
            if len(image.figures) > 0 and (
                len(image.added_figures) == 0 or self.allow_removal_after_insertion
            ):
                # Remove this in case want to depete figures once
                allowed_removes.append(self._get_remove_function())

        if return_inserts_removes:
            return allowed_inserts + allowed_removes, allowed_inserts, allowed_removes
        return allowed_inserts + allowed_removes

    # def get_random_pair(self) -> ImagePair:
    #     pre_figures = np.random.randint(self.grid_size**2 + 1)

    #     target_image = self.init_image()
    #     for _ in range(pre_figures):
    #         target_image = self.add_random_figure(target_image)

    #     target_image.added_figures = []
    #     target_image.removed_figures = []
    #     source_image = target_image.copy()
    #     if self.max_transformations is not None:
    #         max_removes = min(self.max_transformations, pre_figures)
    #     else:
    #         max_removes = pre_figures

    #     remove_figures = np.random.randint(
    #         (1 if pre_figures == (self.grid_size**2) else 0), max_removes + 1
    #     )  # if pre_figures > 0 else 0
    #     for _ in range(remove_figures):
    #         target_image = self.remove_figure(target_image)
    #         # target_image.applied_transformations.append("remove")

    #     empty_slots = self.grid_size**2 - (pre_figures - remove_figures)
    #     if self.max_transformations is not None:
    #         max_inserts = min(self.max_transformations - remove_figures, empty_slots)
    #     else:
    #         max_inserts = empty_slots

    #     post_figures = np.random.randint(
    #         (1 if remove_figures == 0 else 0), max_inserts + 1
    #     )  # if empty_slots > 0 else 0
    #     for _ in range(post_figures):
    #         target_image = self.add_random_figure(target_image)
    #         # target_image.applied_transformations.append("insert")

    #     # print(pre_figures, remove_figures, post_figures)
    #     pair = ImagePair(source_image, target_image)

    #     return pair
