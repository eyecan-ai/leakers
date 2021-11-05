from pathlib import Path
from typing import Dict, Optional, Sequence
import torch
import kornia.augmentation as K
import kornia.geometry.transform as KT
import torch.nn.functional as F
from leakers.datasets.alphabet import BinaryAlphabetDataset
from torch.utils.data import DataLoader
from IPython.display import clear_output
import numpy as np
import cv2
import rich
from leakers.datasets.factory import AlphabetDatasetFactory
from leakers.nn.modules.elastic import ElasticCoder, ElasticDecoder, ElasticEncoder
from leakers.nn.modules.randomizers import VirtualRandomizer
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.core.hooks import DataHooks
import wandb
from typing import Union
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from comet_ml import OfflineExperiment, ExistingExperiment, Experiment
from pytorch_lightning.loggers.base import LoggerCollection
from wandb.sdk.wandb_run import Run
import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import numpy as np
import kornia
from torchvision.utils import make_grid
from matplotlib import cm
import click
import imageio

from leakers.trainers.modules import LeakersTrainingModule


class AlphabetDatamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--output_folder", default="/tmp/leakers")
def train(output_folder: str):

    output_folder = "/tmp/leakers"
    experiment_name = "leaker_alpha"
    epochs = 8000
    checkpoint = ""
    device = "cuda"
    code_size = 5
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
            "type": "virtual",
            "params": {
                "color_jitter": True,
                "random_erasing": True,
            },
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
    datamodule = AlphabetDatamodule(
        dataset=AlphabetDatasetFactory.create(hparams["dataset"]), batch_size=batch_size
    )

    # model = RotPredictor().to(device)
    module = LeakersTrainingModule(**hparams)

    logger = pl_loggers.TensorBoardLogger(output_folder, name=experiment_name)
    trainer = pl.Trainer(
        gpus=-1,
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
