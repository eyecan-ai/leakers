from typing import Union
import cv2
import numpy as np


class ArucoDetector:

    ARUCO_DICTS = {
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

    def __init__(self, aruco_dict: Union[str, tuple] = (8, 3)) -> None:

        if isinstance(aruco_dict, str):
            self._aruco_dict = cv2.aruco.Dictionary_get(
                ArucoDetector.ARUCO_DICTS[aruco_dict]
            )
        elif isinstance(aruco_dict, tuple):
            self._aruco_dict = cv2.aruco.custom_dictionary(*aruco_dict)
        else:
            raise ValueError("Unknown aruco dict type")

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
