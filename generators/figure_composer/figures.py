from typing import Tuple, Union

import numpy as np
import PIL.Image
import PIL.ImageDraw


class Figure:
    COLORS = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "grey": (128, 128, 128),
    }

    def __init__(
        self,
        location: Tuple[int, int],
        size: int,
        rotation: int,
        color: Union[str, Tuple[int, int, int]],
    ):
        self.location = location
        self.size = size
        self.rotation = rotation
        self.color = color

        if isinstance(self.color, str):
            self.color = self.COLORS[self.color]

    def __eq__(self, other: "Figure") -> bool:
        return (
            isinstance(other, self.__class__)
            and self.location == other.location
            and self.size == other.size
            and self.rotation == other.rotation
            and self.color == other.color
        )

    def draw(self, image: PIL.Image.Image) -> PIL.Image.Image:
        raise NotImplementedError()

    @classmethod
    def generate_random(cls, image_size: int) -> "Figure":
        color = np.random.choice(list(cls.COLORS.keys()))
        location = tuple(
            np.random.randint(int(0.2 * image_size), int(0.6 * image_size), 2)
        )
        size = np.random.randint(int(image_size * 0.1), int(image_size * 0.3))
        rotation = np.random.randint(360)
        return cls(location=location, size=size, rotation=rotation, color=color)


class Triangle(Figure):
    def draw(self, image):
        patch = PIL.Image.new("RGBA", (self.size, self.size), (0, 0, 0, 255))
        draw = PIL.ImageDraw.Draw(patch)
        triangle = draw.polygon(
            [(0, self.size - 1), (self.size // 2, 0), (self.size - 1, self.size - 1)],
            fill=self.color + (255,),
        )
        patch = patch.rotate(self.rotation, expand=True, fillcolor=(0, 0, 255, 0))

        location = (
            self.location[0] - patch.width // 2,
            self.location[1] - patch.height // 2,
        )
        image.paste(patch, location, patch)

        return image


class Square(Figure):
    def draw(self, image):
        patch = PIL.Image.new("RGBA", (self.size, self.size), color=self.color + (255,))
        patch = patch.rotate(self.rotation, expand=True, fillcolor=(0, 0, 0, 0))

        location = (
            self.location[0] - patch.width // 2,
            self.location[1] - patch.height // 2,
        )
        image.paste(patch, location, patch)

        return image


class Circle(Figure):
    def draw(self, image):
        patch = PIL.Image.new("RGBA", (self.size, self.size), (0, 0, 0, 255))
        draw = PIL.ImageDraw.Draw(patch)
        circle = draw.ellipse(
            [0, 0, self.size - 1, self.size - 1], fill=self.color + (255,)
        )

        location = (
            self.location[0] - patch.width // 2,
            self.location[1] - patch.height // 2,
        )
        image.paste(patch, location, circle)

        return image
