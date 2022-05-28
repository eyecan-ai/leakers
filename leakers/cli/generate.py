from pathlib import Path
import click


@click.command("generate", help="Compile Configuration file")
@click.option("-c", "--configuration", required=True, help="Configuration File.")
@click.option(
    "-b",
    "--code_size",
    default=3,
    help="Leaker bit size. Number of leakers will be 2^[code_size]",
)
@click.option("-n", "--name", default="leaker_code", help="Leaker generation name")
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

    from leakers.trainers.factory import LeakersConfigurationsBucket
    from choixe.configurations import XConfig
    import rich
    from typing import Dict
    from leakers.datasets.datamodules import GenericAlphabetDatamodule
    from leakers.datasets.factory import AlphabetDatasetFactory
    import pytorch_lightning as pl
    import pytorch_lightning.loggers as pl_loggers
    import click
    from leakers.trainers.modules import LeakersTrainingModule
    import torch
    import shutil

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

    module = LeakersTrainingModule(**hparams)

    # dataset
    datamodule = GenericAlphabetDatamodule(
        dataset=AlphabetDatasetFactory.create(hparams["dataset"]),
        batch_size=batch_size,
        exapansion=100,
    )

    # model = RotPredictor().to(device)
    module = LeakersTrainingModule(**hparams)

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
    saved_model_path = Path(f"{name}.leak")
    shutil.copyfile(checkplint_path, saved_model_path)
    rich.print("Leakers saved to:", saved_model_path)
