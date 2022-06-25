import rich
import torch
import numpy as np
from leakers.detectors.aruco import ArucoDetector
from leakers.detectors.factory import LeakersDetectorsFactory
from leakers.nn.modules.warping import (
    PlugTestConfiguration,
    WarpingPlugTester,
)
import torch.nn.functional as F
from itertools import product
import json
import os


def display_image(img):
    if isinstance(img, torch.Tensor):
        img = (img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return img


def bench_aruco():

    # Parameters
    num_markers = 1024
    num_bits = 4

    # Marker image
    plug_test_configuration = PlugTestConfiguration()

    # Plug tester
    plug_tester = WarpingPlugTester(
        focal_length=plug_test_configuration.focal_length,
        canvas_size=plug_test_configuration.canvas_size,
    )

    # Aruco Detector
    aruco_dict = (num_markers, num_bits)
    aruco_detector = ArucoDetector(aruco_dict=aruco_dict)

    # Testing ids
    if plug_test_configuration.ids is None:
        ids = [num_markers - 1, 0, num_markers // 2]
    else:
        ids = plug_test_configuration.ids

    # Combinations
    raz = list(
        product(
            ids,
            plug_test_configuration.radiuses,
            plug_test_configuration.azimuths,
            plug_test_configuration.zeniths,
        )
    )
    for marker_id, radius, azimuth, zenith in raz:

        marker_img = aruco_detector.generate(
            plug_test_configuration.marker_size, marker_id
        )

        before_detection = aruco_detector.detect(
            marker_img, padding=plug_test_configuration.detection_padding
        )

        assert len(before_detection["ids"]) == 1
        assert before_detection["ids"][0] == marker_id

        # Grayscale ARUCO to x image tensor
        x = (
            torch.tensor(marker_img / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .float()
        )

        out = {
            "aruco_dict": aruco_dict,
            "marker_id": int(marker_id),
            "radius": float(radius),
            "azimuth": float(azimuth),
            "zenith": float(zenith),
            "canvas_size": plug_test_configuration.canvas_size,
            "p": 1,
            "tp": 0,
        }
        results = plug_tester.warp_image(
            x,
            radius=radius,
            azimuth=azimuth,
            zenith=zenith,
        )

        out["area"] = int(results["area"])

        unwarper_image = display_image(results["unwarped_img"])

        # TEST RECALL
        after_detection = aruco_detector.detect(
            unwarper_image,
            padding=plug_test_configuration.detection_padding,
        )

        ids = after_detection["ids"]
        if ids is not None:
            if len(ids) == 1 and ids[0] == marker_id:
                out["tp"] = 1
        ###########################

        with open(f"bench_aruco_{aruco_dict[0]}_{aruco_dict[1]}.json", "a") as f:
            json.dump(out, f)
            f.write(os.linesep)
        rich.print(out)


if __name__ == "__main__":
    bench_aruco()
