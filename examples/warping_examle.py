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
from leakers.nn.modules.warping import WarpingModule, WarpingModuleV2
import torch.nn.functional as F


def display_image(img):
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    return img


def test_square3d():

    # Marker image
    marker_size = [64, 64]
    H, W = 500, 500  # marker_size[0] * 2, marker_size[1] * 2
    mH, mW = marker_size

    random_img = torch.rand(1, 3, 8, 8) * 255.0
    random_img = F.interpolate(
        random_img, size=(marker_size[0], marker_size[1]), mode="nearest"
    )

    # Buffer
    images = []

    K = np.array([2000, 0, W / 2.0, 0, 2000, H / 2.0, 0, 0, 1]).reshape((3, 3))
    K = torch.Tensor(K).unsqueeze(0)
    warper = WarpingModuleV2(camera_matrix=K)

    for radius in [1.0]:
        for azimuth in [0]:  # range(0, 360, 36):
            for zenith in [85]:  # range(0, 90, 20):
                clear_output(True)

                for delta in [0.0]:
                    T_offset = np.array(
                        [
                            [1.0, 0, 0, delta],
                            [0.0, 1, 0, 0],
                            [0.0, 0, 1, 0],
                            [0.0, 0, 0, 1],
                        ]
                    )

                    T = WarpingModuleV2.spherical_marker_transform(
                        radius, azimuth, zenith
                    )
                    T = np.dot(T, T_offset)

                    T = torch.Tensor(T).unsqueeze(0)
                    # display_image(random_img)

                    # try:
                    warped_img = warper.warp_image(
                        random_img, transforms=T, canvas_size=[H, W]
                    )
                    unwarped_img = warper.unwarp_image(
                        warped_img, transforms=T, square_size=[mH, mW]
                    )

                    images.append((unwarped_img, (radius, azimuth, zenith)))
                    # except Exception as e:
                    #     print("ERROR", e)
                    #     pass

                    # display_image(warped_img)
                    # display_image(unwarped_img)
                    # time.sleep(0.2)

            print(len(images))

    for image, pose in images:
        print(pose)
        cv2.imshow("image", display_image(image))
        cv2.waitKey(0)


test_square3d()
