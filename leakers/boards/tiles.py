from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np
from pydantic import BaseModel
import pydantic


class Tile(pydantic.BaseModel, ABC):
    @abstractmethod
    def generate(self, size: int = 64) -> np.ndarray:
        pass


class EmptyTile(Tile):
    color: Tuple[int, int, int] = (255, 255, 255)

    def generate(self, size: int = 64) -> np.ndarray:
        return np.full((size, size, 3), self.color, dtype=np.uint8)


class ImageTile(Tile):
    image: np.ndarray
    padding: int = 10
    border: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _pad(
        self,
        x: np.ndarray,
        padding: int,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):

        return cv2.copyMakeBorder(
            x,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )

    def generate(self, size: int = 64) -> np.ndarray:

        internal_size = size - self.padding * 2 - self.border * 2
        assert internal_size > 0, "Internal size must be greater than 0"

        image = self.image.copy()

        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)

        image = cv2.resize(
            self.image,
            (internal_size, internal_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image = (image * 255).astype(np.uint8)

        # double pad to add Border and Padding
        image = self._pad(
            self._pad(image, self.border, (0, 0, 0)),
            self.padding,
            (255, 255, 255),
        )

        return image


class TriangulatedTile(Tile):

    size: int = 128
    primary_color: Tuple[int, int, int] = (0, 255, 0)
    secondary_color: Tuple[int, int, int] = (255, 0, 0)
    rotation: int = 0

    @property
    def _vertices(self) -> np.ndarray:
        return np.array(
            [
                [0, 0],
                [self.size, 0],
                [self.size, self.size],
                [0, self.size],
            ]
        )

    def _draw_triangle(
        self,
        image: np.ndarray,
        vertex_id: int,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:

        GAL = lambda x: x % 4
        indices = [GAL(vertex_id - 1), vertex_id, GAL(vertex_id + 1)]
        vertices = self._vertices[indices, :]
        pts = vertices.reshape((-1, 1, 2))
        image = cv2.fillPoly(image, [pts], color=color)
        return image

    def generate(self, size: int = 128) -> np.ndarray:
        tile = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        tile = self._draw_triangle(tile, vertex_id=0, color=self.primary_color)
        tile = self._draw_triangle(tile, vertex_id=2, color=self.secondary_color)
        tile = cv2.resize(tile, (size, size))

        for _ in range(self.rotation):
            tile = np.rot90(tile)
        return tile
