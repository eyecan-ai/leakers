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


class AccuracyAccumulator:
    def __init__(self) -> None:
        self._totals_map = {}
        self._corrects_map = {}

    def store(self, name: str, total: int, corrects: int):
        if name not in self._totals_map:
            self._totals_map[name] = 0.0
            self._corrects_map[name] = 0.0

        self._totals_map[name] += total
        self._corrects_map[name] += corrects

    @property
    def scores_map(self):
        omap = {}
        for k, v in self._totals_map.items():
            total = self._totals_map[k]
            corrects = self._corrects_map[k]
            accuracy = corrects / total
            omap[k] = {"total": total, "corrects": corrects, "accuracy": accuracy}
        return omap


class PipelineUtils(object):
    @classmethod
    def merge_outputs(
        cls, outputs: Sequence[Dict[str, torch.Tensor]], key: str
    ) -> torch.Tensor:
        """Merges multiple step outputs

        :param outputs: sequence of dict with single step output. Each single result contains
        a key along with a torch.Tensor with shape [B, * ]
        :type outputs: Sequence[Dict[str, torch.Tensor]]
        :param key: key to merge
        :type key: str
        :return: concat of N single torch.Tensor result, with output shape [BxN, *]
        :rtype: torch.Tensor
        """

        return torch.cat(
            [
                out[key] if len(out[key].shape) > 0 else out[key].unsqueeze(0)
                for out in outputs
            ]
        )

    @classmethod
    def average_outputs(
        cls, outputs: Sequence[Dict[str, torch.Tensor]], key: str
    ) -> torch.Tensor:
        """Mean multiple step outputs values

        :param outputs: sequence of dict with single step output. Each single result contains
        a key along with a torch.Tensor with shape [B]
        :type outputs: Sequence[Dict[str, torch.Tensor]]
        :param key: key to merge
        :type key: str
        :return: concat of N single torch.Tensor result, with output shape [BxN, *]
        :rtype: torch.Tensor
        """

        return torch.mean(torch.Tensor([out[key].item() for out in outputs]))

    @classmethod
    def sum_outputs(
        cls, outputs: Sequence[Dict[str, torch.Tensor]], key: str
    ) -> torch.Tensor:
        """Sum multiple step outputs values

        :param outputs: sequence of dict with single step output. Each single result contains
        a key along with a torch.Tensor with shape [B]
        :type outputs: Sequence[Dict[str, torch.Tensor]]
        :param key: key to merge
        :type key: str
        :return: concat of N single torch.Tensor result, with output shape [BxN, *]
        :rtype: torch.Tensor
        """

        return torch.sum(torch.Tensor([out[key].item() for out in outputs]))


class TensorUtils:
    @classmethod
    def make_grid_of_grids(
        cls,
        images: Sequence[torch.Tensor],
        k: int = -1,
        tile_size: Tuple[int, int] = (128, 128),
        inner_rows: int = -1,
        outer_rows: int = -1,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """Organizes a list of image batches into a unique image.

        :param images: a list of rgb or grayscale image batches
        :type images: Sequence[torch.Tensor]
        :param k: maximum number of images that will be kept for each batch (all images if negative), defaults to -1
        :type k: int, optional
        :param tile_size: height and width of each tilem, defaults to (128, 128)
        :type tile_size: Tuple[int, int], optional
        :param inner_rows: number of rows per inner tile, ceil(sqrt(batch_size)) if negative, defaults to -1
        :type inner_rows: int, optional
        :param outer_rows: number of rows per outer tile, ceil(sqrt(len(images))) if negative, defaults to -1
        :type outer_rows: int, optional
        :param pad_value: padding value, defaults to 0.0
        :type pad_value: float
        :return: output grid image
        :rtype: torch.Tensor
        """

        images = [x.cpu() for x in images]

        if inner_rows < 1:
            inner_rows = math.ceil(math.sqrt(images[0].shape[0]))
        if outer_rows < 1:
            outer_rows = math.ceil(math.sqrt(len(images)))
        if k < 0:
            k = max([x.shape[0] for x in images])

        for i, x in enumerate(images):
            x = torch.cat(
                [pad_value * torch.ones(k - x[0:k].shape[0], *x.shape[1:]), x[0:k]]
            )
            x = kornia.resize(x, tile_size)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = make_grid(x, nrow=inner_rows, padding=0, pad_value=pad_value)
            images[i] = x

        images = torch.stack(images)
        images = make_grid(images, nrow=outer_rows, padding=0, pad_value=pad_value)
        return images

    @staticmethod
    def display_tensors(tensors, max_batch_size: int = -1):
        if max_batch_size == -1:
            max_batch_size = tensors[0].shape[0]
        else:
            max_batch_size = min(max_batch_size, tensors[0].shape[0])

        stacks = []
        for tensor in tensors:
            grid = make_grid(
                tensor[:max_batch_size, ::].detach().cpu(), nrow=max_batch_size
            )
            stacks.append(grid)

        grid = make_grid(stacks, nrow=1)
        return grid


class ImageLogger(object):
    def __init__(self):
        super().__init__()
        print("Image logger built!")

    def log_image(self, name: str, x: torch.Tensor, step: int = -1) -> None:
        """Logs images by name using internal global_step as time
        :param name: log item name
        :type name: str
        :param x: tensor representin image to log ( [3 x H x W] ?)
        :type x: torch.Tensor
        """
        meth_map = {
            SummaryWriter: self._tensorboard_log_image,
            Run: self._wandb_log_image,
        }
        # Multi experiments managed by default
        experiments = (
            self.logger.experiment
            if isinstance(self.logger, LoggerCollection)
            else [self.logger.experiment]
        )
        if step < 0:
            step = self.global_step
        for exp in experiments:
            meth = meth_map.get(type(exp))
            if meth is not None:
                meth(exp, name, x, step)
        # print("LOGGED ", name, self.trainer.log_save_interval)

    def _tensorboard_log_image(
        self, exp: SummaryWriter, name: str, x: torch.Tensor, step: int
    ) -> None:
        x = (x.detach().cpu() * 255).type(torch.uint8)
        exp.add_image(name, x, step)

    def _wandb_log_image(self, exp: Run, name: str, x: torch.Tensor, step: int) -> None:

        data = x.detach().cpu().permute(1, 2, 0).numpy()
        exp.log({name: [wandb.Image(data, caption=name)]}, step=step)


class LeakersTrainingModule(pl.LightningModule, ImageLogger):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.model = ElasticCoder(
            image_shape=self.hparams.image_shape,
            code_size=self.hparams.bit_size,
        )
        self.randomizer = VirtualRandomizer(channel_shuffle=True, channel_shuffle_p=1.0)

        # Losses
        self.weight_code = self.hparams.weight_code
        self.weight_rot = self.hparams.weight_rot
        self.code_loss = torch.nn.SmoothL1Loss()
        self.rot_loss = torch.nn.CrossEntropyLoss()

        self.proto_dataset = BinaryAlphabetDataset(bit_size=self.hparams.bit_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        code = batch["x"]
        B, _ = code.shape

        angles = torch.randint(0, 4, [B]).to(code.device)

        imgs = self.model.generate(code, angles)
        imgs = self.randomizer(imgs)
        out = self.model.encode(imgs)

        code_pred = out["code"]
        rot_logit = out["rot_logits"]

        loss_code = self.code_loss(code, code_pred)
        loss_rot = self.rot_loss(rot_logit, angles)

        loss = self.weight_code * loss_code + self.weight_rot * loss_rot

        return {"loss": loss, "loss_code": loss_code, "loss_rot": loss_rot}

    def debug_display(self, imgs, imgs_t, batch_size: int = 8):
        cols = []
        for b in range(batch_size):
            _img = imgs[b, ::].permute(1, 2, 0).detach().cpu().numpy()
            _img_t = imgs_t[b, ::].permute(1, 2, 0).detach().cpu().numpy()
            cols.append(np.vstack([_img, _img_t]))
        out_img = np.hstack(cols)
        cv2.imshow("output", out_img)
        cv2.waitKey(1)

    def validation_step(self, batch, batch_idx):
        code = batch["x"]
        y = batch["y"]
        B, _ = code.shape

        loss = 0.0
        loss_code = 0.0
        loss_rot = 0.0

        code_corrects = 0.0
        code_total = 0.0
        rot_corrects = 0.0
        rot_total = 0.0

        for rot in [0, 1, 2, 3]:
            angles = torch.Tensor([rot]).repeat(B).to(code.device).long()

            imgs = self.model.generate(code, angles)
            imgs_t = self.randomizer(imgs)

            if batch_idx == 0 and rot == 0:
                self.debug_display(imgs, imgs_t)
                self.log_image(
                    "val/leakers",
                    TensorUtils.display_tensors([imgs, imgs_t], max_batch_size=8),
                )

            out = self.model.encode(imgs_t)

            code_pred = out["code"].detach().cpu().numpy()
            code_idx_pred = self.proto_dataset.words_to_indices(code_pred)
            code_idx = y.cpu().numpy()

            rot_target = angles
            rot_pred = out["rot_classes"]

            code_corrects += (code_idx == code_idx_pred).sum().item()
            code_total += B
            rot_corrects += (rot_target == rot_pred).sum().item()
            rot_total += B

        return {
            "code_corrects": torch.Tensor([code_corrects]),
            "code_total": torch.Tensor([code_total]),
            "rot_corrects": torch.Tensor([rot_corrects]),
            "rot_total": torch.Tensor([rot_total]),
        }

    def validation_epoch_end(self, outputs) -> None:

        code_total = PipelineUtils.sum_outputs(outputs, key="code_total")
        code_corrects = PipelineUtils.sum_outputs(outputs, key="code_corrects")
        rot_total = PipelineUtils.sum_outputs(outputs, key="rot_total")
        rot_corrects = PipelineUtils.sum_outputs(outputs, key="rot_corrects")

        code_accuracy = code_corrects / code_total
        rot_accuracy = rot_corrects / rot_total
        self.log("val/accuracy/code", code_accuracy)
        self.log("val/accuracy/rot", rot_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class AlphabetDatamodule(pl.LightningDataModule):
    def __init__(self, bit_size: int = 6, batch_size: int = 64):
        super().__init__()
        self.bit_size = bit_size
        self.batch_size = batch_size
        self.alphabet_dataset = BinaryAlphabetDataset(bit_size=self.bit_size)

    def train_dataloader(self):
        return DataLoader(
            self.alphabet_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.alphabet_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.alphabet_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


saved_model = None


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--output_folder", default="/tmp/leakers")
def train(output_folder: str):

    output_folder = "/tmp/leakers"
    experiment_name = "leaker_alpha"
    epochs = 1000
    checkpoint = ""
    device = "cuda"

    seed = 211285
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    hparams = {
        "lr": 0.0001,
        "image_shape": [3, 64, 64],
        "bit_size": 6,
        "weight_code": 1.0,
        "weight_rot": 0.001,
    }
    batch_size = hparams["bit_size"] ** 2

    # dataset
    datamodule = AlphabetDatamodule(bit_size=hparams["bit_size"], batch_size=batch_size)

    # model = RotPredictor().to(device)
    module = LeakersTrainingModule(**hparams)

    logger = pl_loggers.TensorBoardLogger(output_folder, name=experiment_name)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        default_root_dir=output_folder,
        check_val_every_n_epoch=10,
        resume_from_checkpoint=checkpoint if len(checkpoint) > 0 else None
        # callbacks=[es]
    )

    trainer.fit(module, datamodule)


@cli.command("export")
@click.option("--checkpoint", required=True)
@click.option("--output_folder", required=True)
def export(checkpoint: str, output_folder: str):

    hparams = {
        "lr": 0.0001,
        "image_shape": [3, 64, 64],
        "bit_size": 6,
        "weight_code": 1.0,
        "weight_rot": 0.001,
    }
    batch_size = hparams["bit_size"] ** 2
    device = "cuda"

    module = LeakersTrainingModule(**hparams).to(device)
    module.load_state_dict(torch.load(checkpoint)["state_dict"])
    module.eval()

    print(module.model)

    # dataset
    datamodule = AlphabetDatamodule(bit_size=hparams["bit_size"], batch_size=batch_size)

    output_images = []
    for sample in datamodule.val_dataloader():

        imgs = module.model.generate(sample["x"].to(device))
        y = sample["y"].to(device)

        B, C, H, W = imgs.shape
        for b in range(B):
            output_images.append(
                (y[b].item(), imgs[b, ::].permute(1, 2, 0).detach().cpu().numpy())
            )

    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    for o in output_images:
        name = f"{str(o[0]).zfill(5)}.png"
        output_file = output_folder / name

        bordersize = 1
        marker = cv2.copyMakeBorder(
            np.uint8(o[1] * 255),
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        imageio.imwrite(output_file, marker)

    from leakers.detectors.simple import SimpleLeakerDetector

    detector = SimpleLeakerDetector(
        model=module.model, dataset=datamodule.val_dataloader().dataset, device="cpu"
    )

    cam = cv2.VideoCapture("http://192.168.1.7:4747/video")
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break

        detections = detector.build_detections(img, size=64)

        for detection in detections:
            rectangle, leaker, _ = detection

            output = detector.detect_single_leaker(leaker)
            if output is not None:
                code, rot = output["code"], output["rot"]

                rich.print("Detection", code, rot)

                points = rectangle.reshape((-1, 2))
                points = np.roll(points, rot, axis=0)
                corner = points[0, :]

                cv2.drawContours(img, [rectangle], 0, (255, 255, 0), 3)
                cv2.circle(img, tuple(corner), 5, (0, 0, 255), -1)
                text = f"ID: {code}"
                cv2.putText(
                    img,
                    text,
                    tuple(np.int32(corner - np.array([0, 20]))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("cam", img)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    cli()
