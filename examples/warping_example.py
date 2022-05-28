from typing import List, Sequence, Optional
import cv2
from PIL import Image
import rich
import transforms3d
import kornia.augmentation as kaug
import time
import imageio
import torch
import numpy as np
from leakers.nn.modules.warping import WarpingModule
import torch.nn.functional as F

from leakers.utils import TransformsUtils


def detect_rectangles(input_image: np.ndarray) -> Sequence[np.ndarray]:
    """Detects rectangles in the image.

    :param input_image: input image
    :type input_image: np.ndarray
    :return: detected rectangles
    :rtype: Sequence[np.ndarray]
    """
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, hy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Searching through every region selected to
    # find the required polygon.
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        min_max_area_size = [1, 600 * 400]
        eps_curve = 0.1
        # Shortlisting the regions based on there area.
        if area > min_max_area_size[0] and area < min_max_area_size[1]:
            approx = cv2.approxPolyDP(cnt, eps_curve * cv2.arcLength(cnt, True), True)
            print("\tSIZE", len(approx))
            if len(approx) == 4:
                results.append(approx)
    return results


CVMarkerDICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class CVMarker:
    def __init__(self, aruco_dict: str = "DICT_4X4_50", size: int = 128) -> None:
        self._size = size
        self._dict = cv2.aruco.Dictionary_get(CVMarkerDICTS[aruco_dict])

    @property
    def aruco_dict(self) -> int:
        return self._dict

    def draw(self) -> np.ndarray:
        return cv2.aruco.drawMarker(self._dict, 0, self._size)


class ArucoDetector:
    def __init__(self, aruco_dict: str = "DICT_4X4_50") -> None:
        self._aruco_dict = cv2.aruco.Dictionary_get(CVMarkerDICTS[aruco_dict])

    def detect(self, image: np.ndarray, padding: int = 0) -> np.ndarray:
        if padding > 0:
            image = cv2.copyMakeBorder(
                image,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=255,
            )

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            image,
            self._aruco_dict,
        )
        return {"corners": corners, "ids": ids}

    def generate(self, size: int, id: int) -> np.ndarray:
        return cv2.aruco.drawMarker(self._aruco_dict, id, size)


def display_image(img):
    if isinstance(img, torch.Tensor):
        img = (img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return img


class MarkerPlugTester:
    def __init__(
        self,
        focal_length: float,
        canvas_size: List[int] = [500, 500],
    ) -> None:

        # virtual canvas size
        self._canvas_size = canvas_size
        H, W = self._canvas_size

        # camera matrix
        self._K = (
            torch.tensor(
                np.array(
                    [
                        [focal_length, 0, W / 2.0],
                        [0, focal_length, H / 2.0],
                        [0, 0, 1],
                    ]
                )
            )
            .unsqueeze(0)
            .float()
        )

        # image warper
        self._warper = WarpingModule(camera_matrix=self._K)

    def spherical_transform(
        self,
        radius: float = 1.0,
        azimuth: float = 45.0,
        zenith: float = 45.0,
    ) -> torch.Tensor:
        T_offset = TransformsUtils.translation_transform(0.0, 0.0, 0.0)
        T = WarpingModule.spherical_marker_transform(radius, azimuth, zenith)
        T = np.dot(T, T_offset)
        T = torch.Tensor(T).unsqueeze(0)
        return T

    def warp_image(
        self,
        x: torch.Tensor,
        radius: float = 1.0,
        azimuth: float = 45.0,
        zenith: float = 45.0,
        background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # virtual canvas size
        mH, mW = x.shape[2:]
        H, W = self._canvas_size

        if background is None:
            background = torch.ones((1, 3, H, W)).float()

        mask = torch.ones_like(x)

        T = self.spherical_transform(radius, azimuth, zenith)

        # warp mask
        warped_mask = self._warper.warp_image(
            mask,
            transforms=T,
            canvas_size=[H, W],
            mode="bilinear",
        )

        # warp input image
        warped_x = self._warper.warp_image(
            x,
            transforms=T,
            canvas_size=[H, W],
            mode="bilinear",
        )

        # compose virtual image with background
        virtual_image = warped_x * warped_mask + background * (1 - warped_mask)

        # wark image back
        unwarped_img = self._warper.unwarp_image(
            warped_x,
            transforms=T,
            square_size=[mH, mW],
            mode="bilinear",
        )

        return {
            "virtual_image": virtual_image,
            "unwarped_img": unwarped_img,
            "warped_image": warped_x,
            "warped_mask": warped_mask,
        }


def test_square3d():

    # Marker image
    detection_padding = 10
    canvas_size = [500, 500]
    marker_size = [64, 64]
    plug_tester = MarkerPlugTester(focal_length=1000, canvas_size=canvas_size)
    aruco_dict = "DICT_ARUCO_ORIGINAL"
    aruco_detector = ArucoDetector(aruco_dict=aruco_dict)

    marker_img = aruco_detector.generate(marker_size[0], 0)

    results = aruco_detector.detect(marker_img, padding=detection_padding)
    rich.print(results)
    cv2.imshow("marker", marker_img)
    cv2.waitKey(0)

    x = (
        torch.tensor(marker_img / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(1, 3, 1, 1)
        .float()
    )

    results = plug_tester.warp_image(x, 9, 0, 0)

    virtual_image = results["virtual_image"]
    warped_image = results["warped_image"]
    warped_mask = results["warped_mask"]
    unwarped_img = results["unwarped_img"]

    detections = aruco_detector.detect(display_image(unwarped_img), detection_padding)
    print(detections)

    cv2.imshow("virtual_image", display_image(virtual_image))
    cv2.imshow("unwarped_img", display_image(unwarped_img))
    cv2.waitKey(0)


test_square3d()
