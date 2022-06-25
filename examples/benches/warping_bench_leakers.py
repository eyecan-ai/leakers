import rich
import torch
import numpy as np
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


def bench_leakers():

    # Parameters
    leakers_filaneme = "../codeformer/leaker_code.leak"
    device = "cpu"

    # Marker image
    plug_test_configuration = PlugTestConfiguration()

    # Plug tester
    plug_tester = WarpingPlugTester(
        focal_length=plug_test_configuration.focal_length,
        canvas_size=plug_test_configuration.canvas_size,
    )

    # Leakers Detector
    detector = LeakersDetectorsFactory.create_from_checkpoint(
        filename=leakers_filaneme, device=device
    )

    # Leakers
    raw_leakers = detector.generate_leakers(
        output_size=plug_test_configuration.marker_size,
        border=1,
        padding=0,
    )
    leakers = {}
    for raw_leaker in raw_leakers:
        leakers[raw_leaker["id"]] = raw_leaker["image"]

    # Leakers parameters
    num_markers = detector.model.alphabet_size()
    num_bits = detector.model.code_size()
    dictionary = (num_markers, num_bits)

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

        marker_img = (leakers[marker_id] * 255).astype(np.uint8)
        before_detection = detector.detect(
            marker_img,
            padding=plug_test_configuration.detection_padding,
        )

        codes = {x["code"]: x for x in before_detection}
        assert len(codes) >= 1
        assert marker_id in codes

        x = torch.tensor(marker_img / 255.0).permute(2, 0, 1).unsqueeze(0).float()

        out = {
            "aruco_dict": dictionary,
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

        # Test Recall
        after_detection = detector.detect(
            unwarper_image,
            padding=plug_test_configuration.detection_padding,
        )
        after_codes = {x["code"]: x for x in after_detection}
        if len(after_codes) >= 1:
            if marker_id in after_codes:
                out["tp"] = 1
        ###########################

        with open(f"bench_leakers_{dictionary[0]}_{dictionary[1]}.json", "a") as f:
            json.dump(out, f)
            f.write(os.linesep)
        rich.print(out)


if __name__ == "__main__":
    bench_leakers()
