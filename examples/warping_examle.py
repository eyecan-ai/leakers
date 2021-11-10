import cv2
from PIL import Image
from IPython.display import display
import transforms3d
import kornia.augmentation as kaug
from IPython.display import clear_output
import time
import imageio
from IPython.display import HTML
from IPython.display import Image as IPImage
import torch
import numpy as np
from leakers.nn.modules.warping import WarpingModule
import torch.nn.functional as F

from leakers.utils import TransformsUtils


def display_image(img):
    if isinstance(img, torch.Tensor):
        img = (img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return img


def test_square3d():

    # Marker image
    marker_size = [64, 64]
    H, W = 500, 500  # marker_size[0] * 2, marker_size[1] * 2
    mH, mW = marker_size

    x = torch.rand(1, 3, 8, 8)
    x = F.interpolate(x, size=(marker_size[0], marker_size[1]), mode="nearest")
    mask = torch.ones_like(x)

    background = imageio.imread(
        "/home/daniele/Downloads/gettyimages-1124517056-612x612.jpg"
    )
    background = torch.Tensor(background / 255.0).permute(2, 0, 1).unsqueeze(0)
    background = F.interpolate(background, size=(H, W), mode="nearest")

    # Buffer
    images = []

    K = np.array([2000, 0, W / 2.0, 0, 2000, H / 2.0, 0, 0, 1]).reshape((3, 3))
    K = torch.Tensor(K).unsqueeze(0)
    warper = WarpingModule(camera_matrix=K)

    radius = 1.0
    azimuth = 45.0
    zenith = 75.0
    T_offset = TransformsUtils.translation_transform(0.0, 0.0, 0.0)
    T = WarpingModule.spherical_marker_transform(radius, azimuth, zenith)
    T = np.dot(T, T_offset)

    T = torch.Tensor(T).unsqueeze(0)
    # display_image(random_img)

    # try:
    warped_mask = warper.warp_image(
        mask, transforms=T, canvas_size=[H, W], mode="bilinear"
    )
    warped_x = warper.warp_image(x, transforms=T, canvas_size=[H, W], mode="bilinear")
    unwarped_img = warper.unwarp_image(
        x, transforms=T, square_size=[mH, mW], mode="bilinear"
    )

    print(warped_x.shape, background.shape)
    out = warped_x * warped_mask + background * (1 - warped_mask)
    cv2.imshow("image", display_image(out))
    cv2.waitKey(0)


test_square3d()
