import sys
from typing import Tuple
import click


@click.group("debug")
def debug():
    pass


@debug.command("mosaic", help="Debug Leakers in a Mosaic fashon")
@click.option("-m", "--model", required=True, help="Leakers Model File.")
@click.option("-r", "--display_rows", default=2, help="leakers mosaic rows")
@click.option("-o", "--output_file", default="", help="output file")
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU")
def mosaic(
    model: str,
    display_rows: int,
    output_file: str,
    cuda: bool,
):

    debug_show = len(output_file) == 0

    import cv2
    import cv2
    from leakers.detectors.factory import LeakersDetectorsFactory
    import rich
    import numpy as np
    from einops import rearrange

    device = "cuda" if cuda else "cpu"
    detector = LeakersDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )
    leakers = detector.generate_leakers(
        border=0,
        padding=10,
        output_size=256,
    )

    leakers_images = np.array([leaker["image"] for leaker in leakers])
    leakers_images = rearrange(
        leakers_images,
        "(bH bW) h w c -> (bH h) (bW w) c",
        bH=display_rows,
    )
    leakers_images = (leakers_images * 255).astype(np.uint8)

    if debug_show:
        cv2.namedWindow(f"whole", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(f"whole", cv2.cvtColor(leakers_images, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0 if debug_show else 1)

    detections = detector.detect(leakers_images)
    rich.print(detections)
    for detection in detections:
        leakers_images = detector.draw_detection(leakers_images, detection)

    if debug_show:
        cv2.imshow(f"whole", cv2.cvtColor(leakers_images, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    if not debug_show:
        cv2.imwrite(output_file, cv2.cvtColor(leakers_images, cv2.COLOR_RGB2BGR))
        rich.print(f"Saved to {output_file}")


@debug.command("orbiter", help="Debug Leakers orbiting around them")
@click.option("-m", "--model", required=True, help="Leakers Model File.")
@click.option("-i", "--leaker_ids", nargs=2, default=(0, 0), help="target leakerd ID")
@click.option("-d", "--leaker_distance", default=0.4, help="leaker virtual distance")
@click.option("-a", "--azimuths", nargs=2, default=(-180.0, 180), help="azimuths range")
@click.option("-z", "--zeniths", nargs=2, default=(-89.0, 89.0), help="zeniths range")
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU")
def orbiter(
    model: str,
    leaker_ids: int,
    leaker_distance: float,
    azimuths: Tuple[float, float],
    zeniths: Tuple[float, float],
    cuda: bool,
):

    import cv2
    import cv2
    from leakers.detectors.factory import LeakersDetectorsFactory
    import rich
    import numpy as np
    from einops import rearrange
    from leakers.nn.modules.warping import (
        PlugTestConfiguration,
        WarpingPlugTester,
    )
    import torch

    device = "cuda" if cuda else "cpu"
    detector = LeakersDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )
    leakers = detector.generate_leakers(
        border=1,
        padding=0,
        output_size=256,
    )

    # Marker image
    plug_test_configuration = PlugTestConfiguration()

    # Plug tester
    plug_tester = WarpingPlugTester(
        focal_length=plug_test_configuration.focal_length,
        canvas_size=plug_test_configuration.canvas_size,
    )

    zeniths = np.arange(zeniths[0], zeniths[1], 1)
    azimuths = np.arange(azimuths[0], azimuths[1] + 1, 1)
    leaker_ids = range(leaker_ids[0], leaker_ids[1] + 1)

    for leaker_id in leaker_ids:
        for zenith in zeniths:
            for azimuth in azimuths:
                rich.print(
                    f"Current -> ID[{leaker_id}], Zenith[{zenith}], Azimuth[{azimuth}]"
                )
                leaker = leakers[leaker_id]["image"].copy()

                x = torch.tensor(leaker).permute(2, 0, 1).unsqueeze(0).float()
                results = plug_tester.warp_image(
                    x,
                    radius=leaker_distance,
                    azimuth=azimuth,
                    zenith=zenith,
                )

                output_image = (
                    results["virtual_image"][0, ::].detach().permute(1, 2, 0).numpy()
                )

                output_image = (output_image * 255).astype(np.uint8)

                # detect the leaker
                detections = detector.detect(output_image)
                rich.print(detections)
                for detection in detections:
                    output_image = detector.draw_detection(output_image, detection)

                cv2.imshow(f"whole", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(0)
                if k == ord("q"):
                    sys.exit(0)
