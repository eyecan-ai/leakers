from typing import List, Tuple
import cv2
from einops import rearrange
import numpy as np
import pydantic
from leakers.boards.tiles import EmptyTile, ImageTile, Tile, TriangulatedTile


class SimpleBoard(pydantic.BaseModel):
    tiles: List[Tile] = []
    width: int = 10
    height: int = 10
    padding: int = 32

    def generate(self, tile_size: int = 64) -> np.ndarray:
        assert len(self.tiles) == self.width * self.height

        images = [tile.generate(tile_size) for tile in self.tiles]
        images = np.array(images)
        board_image = rearrange(
            images,
            "(bH bW) h w c -> (bH h) (bW w) c",
            bH=self.height,
        )

        board_image = cv2.copyMakeBorder(
            board_image,
            top=self.padding,
            bottom=self.padding,
            left=self.padding,
            right=self.padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        return board_image


class RibbonImagesPoolBoard(SimpleBoard):

    first_color: Tuple[int, int, int] = (0, 0, 0)
    second_color: Tuple[int, int, int] = (173, 20, 87)
    third_color: Tuple[int, int, int] = (0, 230, 118)
    images: List[np.ndarray] = []
    images_padding_size: int = 0
    images_border_size: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tiles.clear()
        imgs_pool_counter = 0
        for row in range(self.height):
            for col in range(self.width):
                state = (col + row) % 2
                if state == 0:
                    if row % 2 == 0:
                        tile = TriangulatedTile(
                            rotation=0,
                            primary_color=self.first_color,
                            secondary_color=self.second_color,
                        )
                    else:
                        tile = TriangulatedTile(
                            rotation=2,
                            primary_color=self.first_color,
                            secondary_color=self.third_color,
                        )
                else:
                    if len(self.images) == 0:
                        tile = EmptyTile()
                    else:
                        image = self.images[imgs_pool_counter % len(self.images)]
                        imgs_pool_counter += 1
                        tile = ImageTile(
                            image=image,
                            padding=self.images_padding_size,
                            border=self.images_border_size,
                        )
                self.tiles.append(tile)
