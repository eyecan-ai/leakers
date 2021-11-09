from typing import Dict
from leakers.datasets.datamodules import GenericAlphabetDatamodule
from leakers.datasets.factory import AlphabetDatasetFactory
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import click
from leakers.trainers.modules import LeakersTrainingModule
import torch


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--output_folder", default="/tmp/leakers")
@click.option("--epochs", default=8000)
def train(output_folder: str, epochs: int):

    output_folder = "/tmp/leakers"
    experiment_name = "leaker_alpha"
    checkpoint = ""
    device = "cpu"
    code_size = 3
    batch_size = 2 ** code_size

    hparams = {
        "coder": {
            "type": "elastic",
            "params": {
                "image_shape": [3, 64, 64],
                "code_size": code_size,
                "cin": 32,
                "n_layers": 4,
                "k": 3,
                "bn": False,
                "act_middle": "torch.nn.ReLU",
                "act_latent": None,
                "act_last": "torch.nn.Sigmoid",
            },
        },
        "randomizer": {
            "type": "warping",
            "params": {"color_jitter": True, "random_erasing": True, "warper": True},
        },
        "dataset": {"type": "binary", "params": {"bit_size": code_size}},
        "losses": {
            "code_loss": "torch.nn.SmoothL1Loss",
            "code_loss_weight": 1.0,
            "rot_loss_weight": 0.1,
        },
        "rotations": {"randomize": False},
        "optimizer": {
            "lr": 0.0001,
        },
    }

    module = LeakersTrainingModule(**hparams)

    # dataset
    datamodule = GenericAlphabetDatamodule(
        dataset=AlphabetDatasetFactory.create(hparams["dataset"]), batch_size=batch_size
    )

    # model = RotPredictor().to(device)
    module = LeakersTrainingModule(**hparams)

    logger = pl_loggers.TensorBoardLogger(output_folder, name=experiment_name)
    torch.autograd.set_detect_anomaly(True)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        default_root_dir=output_folder,
        check_val_every_n_epoch=100,
        resume_from_checkpoint=checkpoint if len(checkpoint) > 0 else None
        # callbacks=[es]
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    cli()


# shutdown computer now
