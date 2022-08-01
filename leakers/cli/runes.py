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
    rich.print(hparams)

    batch_size = 2**code_size

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
    from leakers.detectors.factory import RunesDetectorsFactory
    import rich
    import numpy as np
    from einops import rearrange

    device = "cuda" if cuda else "cpu"
    detector = RunesDetectorsFactory.create_from_checkpoint(
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

    def cv_extract_roi_given_point(image, center, size):
        x, y = center
        image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT)
        x += size
        y += size
        crop = image[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]

        return crop

    def cv_mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            crop = cv_extract_roi_given_point(
                image=leakers_images, center=(x, y), size=256
            )

            detections = detector.detect_single_leaker(crop)
            code = detections["code"]

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
            cv2.waitKey(1)

    cv2.namedWindow(f"debug", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(f"debug", cv_mouse_callback)
    while True:

        # if debug_show:
        cv2.imshow(f"debug", cv2.cvtColor(leakers_images, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
