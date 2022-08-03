import cv2
import numpy as np
from leakers.boards.simple import RibbonImagesPoolBoard, SimpleBoard
from leakers.boards.tiles import TriangulatedTile, EmptyTile


# tile = TriangulatedTile(rotation=0)
tile = EmptyTile()

w = 12
h = 8

tiles = []

images = [np.random.uniform(0, 1, (8, 8, 3)).astype(np.float32) for _ in range(100)]
images = [cv2.resize(img, (64, 64)) for img in images]

board = RibbonImagesPoolBoard(width=w, height=h, images=images)
image = board.generate()
cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
