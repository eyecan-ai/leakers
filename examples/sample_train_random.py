from typing import Sequence
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


saved_model = None


def train_leaker_module():

    device = "cuda"
    bit_size = 6
    batch_size = bit_size ** 2
    batch_size_test = 16

    image_shape = [3, 64, 64]
    image_repeats = 3 if image_shape[0] == 1 else 1

    weight_code = 1.0
    weight_rot = 0.0001

    seed = 1111
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
    model = ElasticCoder(
        image_shape=image_shape,
        code_size=bit_size,
    ).to(device)

    randomizer = VirtualRandomizer()

    """
    Optimizer
    """
    lr = 0.0001

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
            imgs = randomizer(imgs)
            out = model.encode(imgs)
            xh = out["code"]
            rot_logit = out["rot_logits"]

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
                imgs_t = randomizer(imgs)
                out = model.encode(imgs)
                xh = out["code"]
                rot_logit = out["rot_logits"]
                # print("SHAP",imgs.shape)
                # print(x)
                # print(xh)

                cols = []
                for b in range(batch_size_test):
                    _img = (
                        imgs[b, ::]
                        .permute(1, 2, 0)
                        .repeat(1, 1, image_repeats)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    _img_t = (
                        imgs_t[b, ::]
                        .permute(1, 2, 0)
                        .repeat(1, 1, image_repeats)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    print("SHA", _img.shape, _img_t.shape)
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
                    out = model.encode(imgs)
                    xh = out["code"]
                    rot_logit = out["rot_logits"]

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
