import click


@click.group()
def runes():
    pass


@runes.command("generate", help="Compile Configuration file")
@click.option("-c", "--configuration", required=True, help="Configuration File.")
@click.option(
    "-b",
    "--code_size",
    default=3,
    help="Leaker bit size. Number of leakers will be 2^[code_size]",
)
@click.option("-n", "--name", default="rune_code", help="Leaker generation name")
@click.option("--image_size", default=64, help="Leaker size")
@click.option("--channels", default=3, help="Leaker image channels [1 or 3]")
@click.option("-o", "--output_folder", default="/tmp/leakers", help="Output log folder")
@click.option("-e", "--epochs", default=5000, help="Number of epochs")
@click.option("--device", default="cpu", help="Device")
def generate(
    configuration: str,
    code_size: int,
    name: str,
    image_size: int,
    channels: int,
    output_folder: str,
    epochs: int,
    device: str,
):

    from choixe.configurations import XConfig
    import rich
    from typing import Dict
    from leakers.datasets.datamodules import GenericAlphabetDatamodule
    from leakers.datasets.factory import AlphabetDatasetFactory
    import pytorch_lightning as pl
    import pytorch_lightning.loggers as pl_loggers
    from leakers.trainers.modules import RuneTrainingModule
    import torch
    import shutil
    from pathlib import Path

    # Load configuration
    configuration: XConfig = XConfig(filename=configuration)

    # Replace placeholders
    configuration.replace_variables_map(
        {
            "channels": channels,
            "image_size": image_size,
            "code_size": code_size,
        }
    )
    configuration.check_available_placeholders(close_app=True)
    hparams = configuration.to_dict()

    batch_size = min(2 ** code_size, 1024)

    module = RuneTrainingModule(**hparams)

    # dataset
    datamodule = GenericAlphabetDatamodule(
        dataset=AlphabetDatasetFactory.create(hparams["dataset"]),
        batch_size=batch_size,
        exapansion=100,
        num_workers=2,
    )

    logger = pl_loggers.TensorBoardLogger(output_folder, name=name)
    torch.autograd.set_detect_anomaly(True)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        default_root_dir=output_folder,
        check_val_every_n_epoch=1,
        # val_check_interval=0.1
        # resume_from_checkpoint=checkpoint if len(checkpoint) > 0 else None,
    )

    trainer.fit(module, datamodule)

    checkplint_path = trainer.checkpoint_callback.best_model_path
    saved_model_path = Path(f"{name}.rune")
    shutil.copyfile(checkplint_path, saved_model_path)
    rich.print("Runes saved to:", saved_model_path)


@runes.group("debug")
def debug():
    pass


@debug.command("image", help="Load a runes image for interactive debug")
@click.option("-m", "--model", required=True, help="Runes Model File.")
@click.option("-i", "--input_image", default="", help="Leakers Model File.")
@click.option("-w", "--width", default=12, help="Runes Board width.")
@click.option("-h", "--height", default=8, help="Runes Board height.")
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU")
def image(
    model: str,
    input_image: str,
    width: int,
    height: int,
    cuda: bool,
):

    import cv2
    from leakers.detectors.factory import RunesDetectorsFactory
    from leakers.boards.simple import RibbonImagesPoolBoard
    import rich
    import numpy as np
    from einops import rearrange
    import itertools
    import imageio

    device = "cuda" if cuda else "cpu"
    detector = RunesDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )

    if len(input_image) > 0:
        image = imageio.imread(input_image)[:, :, :3]
    else:
        rich.print(f"Generating board {width}x{height}")
        leakers = detector.generate_leakers(
            border=0,
            padding=0,
            output_size=256,
            batch_size=16,
        )
        leakers_images = [leaker["image"] for leaker in leakers]

        board = RibbonImagesPoolBoard(
            width=width,
            height=height,
            images=leakers_images,
        )
        image = board.generate()

    crop_size = 256

    click_points = []

    def cv_mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)

            if len(click_points) == 4:
                click_points.clear()
            else:
                click_points.append(point)

    cv2.namedWindow(f"debug", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(f"debug", cv_mouse_callback)
    while True:

        output_image = image.copy()

        for click_point in click_points:
            output_image = cv2.circle(output_image, click_point, 5, (0, 0, 255), -1)

        if len(click_points) == 4:
            size = detector.model.image_shape()[-1]
            points = np.array(click_points).reshape((4, 2)).astype(np.float32)
            dst = np.array(
                [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(points, dst)
            M2 = cv2.getPerspectiveTransform(dst, points)
            warp = cv2.warpPerspective(image, M, (size, size))
            padding = 3
            warp = warp[padding:-padding, padding:-padding]

            crop = cv2.resize(warp, (size, size), interpolation=cv2.INTER_CUBIC)

            detections = detector.detect_single_leaker(crop)
            code = detections["code"]

            crop = cv2.resize(crop, (256, 256))
            crop = cv2.putText(
                crop,
                f"ID: {code}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            rich.print(detections)

            cv2.imshow("crop", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # if debug_show:
        cv2.imshow(f"debug", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("d"):
            crop_size += 1
        if k == ord("a"):
            crop_size -= 1


@debug.command("print_board", help="Print runes board")
@click.option("-m", "--model", required=True, help="Leakers Model File.")
@click.option("-o", "--output_file", default="", help="output file")
@click.option("-w", "--width", default=12, help="board width")
@click.option("-h", "--height", default=8, help="board height")
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU")
def print_board(
    model: str,
    output_file: str,
    width: int,
    height: int,
    cuda: bool,
):

    import cv2
    from leakers.detectors.factory import RunesDetectorsFactory
    from leakers.boards.simple import RibbonImagesPoolBoard
    import rich

    device = "cuda" if cuda else "cpu"
    detector = RunesDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )
    leakers = detector.generate_leakers(
        border=0,
        padding=0,
        output_size=256,
        batch_size=256,
    )
    leakers_images = [leaker["image"] for leaker in leakers]

    board = RibbonImagesPoolBoard(
        width=width,
        height=height,
        images=leakers_images,
    )
    board_image = board.generate()

    cv2.imshow("board", cv2.cvtColor(board_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    if len(output_file) > 0:
        cv2.imwrite(output_file, cv2.cvtColor(board_image, cv2.COLOR_RGB2BGR))
        rich.print("Board saved to:", output_file)
