from typing import Sequence
import torch
import kornia.augmentation as K
import kornia.geometry.transform as KT
import torch.nn.functional as F
from leakers.datasets.alphabet import BinaryAlphabetDataset
from leakers.nn.modules.base import BasicDecoder, BasicEncoder, Identity
from torch.utils.data import DataLoader
from IPython.display import clear_output
import numpy as np
import cv2
import rich
from leakers.nn.modules.encoders.elastic import ElasticDecoder, ElasticEncoder

from leakers.nn.modules.shufflenet import ShuffleNet


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


class LeakerModule(torch.nn.Module):
    def __init__(
        self, bit_size=8, image_size=[256, 256], channels=3, num_layers: int = 4
    ):
        super().__init__()
        self._bit_size = bit_size
        self._image_size = image_size
        self._channels = channels

        # self._generator = BasicDecoder(
        #     self._image_size,
        #     output_channels=self._channels,
        #     latent_size=self._bit_size,
        #     n_layers=num_layers,
        #     with_batch_norm=False,
        # )

        # self._encoder = BasicEncoder(
        #     input_size=self._image_size,
        #     input_channels=self._channels,
        #     latent_size=self._bit_size + 4,
        #     n_layers=num_layers,
        #     with_batch_norm=False,
        # )
        self._generator = ElasticDecoder(
            output_shape=[self._channels, self._image_size[0], self._image_size[1]],
            latent_size=self._bit_size,
            n_layers=num_layers,
            bn=False,
        )

        self._encoder = ElasticEncoder(
            input_shape=[self._channels, self._image_size[0], self._image_size[1]],
            latent_size=self._bit_size + 4,
            n_layers=num_layers,
            bn=False,
            act_last=None,
        )

        self._randomizer = torch.nn.Sequential(
            # K.ColorJitter(0.2, 0.2, 0.2, hue=0),
            K.RandomCrop([58, 58]),
            torch.nn.Upsample([64, 64]),
            # K.RandomChannelShuffle(p=1.0),
            K.RandomGaussianNoise(mean=0.0, std=0.05),
            K.RandomAffine(
                15, translate=torch.Tensor([0.1, 0.1]), scale=torch.Tensor([0.9, 1.1])
            ),
            K.RandomBoxBlur(kernel_size=[15, 15]),
            K.RandomErasing(scale=(0.02, 0.1), ratio=(0.8, 1.2)),
        )
        # self._randomizer = Identity()

    def generate(self, z, rotate_classes: Sequence[int]):
        img = self._generator(z)
        B, C, H, W = img.shape
        return KT.rotate(img, 90.0 * rotate_classes)

    def encode(self, x):
        out = self._encoder(x)
        z = torch.tanh(out[:, : self._bit_size])
        rot_logit = out[:, self._bit_size : self._bit_size + 4]
        return z, rot_logit

    def randomize(self, x):
        x = self._randomizer(x)
        # x[:,:,:10,:] = 0
        return x


saved_model = None


def train_leaker_module():

    device = "cuda"
    bit_size = 6
    batch_size = bit_size ** 2
    batch_size_test = 16

    channels = 3
    out_repeats = 3 if channels == 1 else 1
    image_size = [64, 64]

    weight_code = 1.0
    weight_rot = 0.01

    seed = 666
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # dataset
    dataset = BinaryAlphabetDataset(bit_size=bit_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dataloader_test = DataLoader(
        dataset, batch_size=batch_size_test, shuffle=False, drop_last=True
    )

    # model = RotPredictor().to(device)
    model = LeakerModule(
        bit_size=bit_size, channels=channels, image_size=image_size
    ).to(device)

    """
    Optimizer
    """
    lr = 0.001

    lr_epochs = {
        100: 0.5,
        200: 0.5,
        300: 0.5,
        400: 0.5,
        500: 0.5,
    }

    rich.print(
        "[red] NUm Parameters: [/red]",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    code_loss = torch.nn.SmoothL1Loss()
    rot_loss = torch.nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(10000):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()

            loss_code = 0.0
            loss_rot = 0.0
            x = batch["x"].to(device)
            B, _ = x.shape

            angles = torch.randint(0, 4, [B]).to(x.device)
            imgs = model.generate(x, angles)
            # print("MINMAX", imgs.min(), imgs.max())
            imgs = model.randomize(imgs)
            xh, rot_logit = model.encode(imgs)
            loss_code += code_loss(x, xh)

            # rot_target = torch.Tensor([rot]).repeat(B).long().to(device)
            # print(rot_logit.shape, rot_target.shape, rot_target)
            loss_rot += rot_loss(rot_logit, angles)

            loss = weight_code * loss_code + weight_rot * loss_rot
            loss.backward()
            optimizer.step()

        if epoch in lr_epochs:
            multiplier = lr_epochs[epoch]
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * multiplier
                print("LR CHANGE ", "**" * 10, g["lr"])

        if epoch % 1 == 0:
            model.eval()

            for batch in dataloader_test:
                x = batch["x"].to(device)
                B, _ = x.shape
                angles = torch.Tensor([0]).repeat(B).to(x.device)
                imgs = model.generate(x, angles)
                imgs_t = model.randomize(imgs)
                xh = model.encode(imgs_t)
                # print("SHAP",imgs.shape)
                # print(x)
                # print(xh)

                cols = []
                for b in range(batch_size_test):
                    _img = (
                        imgs[b, ::]
                        .permute(1, 2, 0)
                        .repeat(1, 1, out_repeats)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    _img_t = (
                        imgs_t[b, ::]
                        .permute(1, 2, 0)
                        .repeat(1, 1, out_repeats)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cols.append(np.vstack([_img, _img_t]))
                out_img = np.hstack(cols)
                cv2.imshow("output", out_img)
                cv2.waitKey(1)

                break

            accumulator = AccuracyAccumulator()
            for batch in dataloader_test:
                for rot in [0, 1, 2, 3]:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    B, _ = x.shape
                    angles = torch.Tensor([rot]).repeat(B).to(x.device)
                    imgs = model.generate(x, angles)
                    # imgs_t = model.randomize(imgs)
                    xh, rot_logit = model.encode(imgs)
                    rot_pred = torch.argmax(rot_logit, dim=1)
                    rot_target = angles

                    code_pred = torch.tanh(xh * 100000).detach().cpu().numpy()
                    code_idx_pred = dataset.words_to_indices(code_pred)
                    code_idx = y.cpu().numpy()

                    accumulator.store(
                        "code",
                        total=B,
                        corrects=(code_idx == code_idx_pred).sum().item(),
                    )

                    accumulator.store(
                        "rot",
                        total=B,
                        corrects=(rot_target == rot_pred).sum().item(),
                    )

            print(
                "Epoch",
                epoch,
                "Loss",
                loss.item(),
                "Loss Code:",
                loss_code.item(),
                "Loss Rot:",
                loss_rot.item(),
            )
            rich.print(accumulator.scores_map)
        # time.sleep(2)
        # break


train_leaker_module()
