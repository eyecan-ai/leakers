from typing import Dict, Sequence, Tuple
import numpy as np
import cv2
import torch
from leakers.datasets.alphabet import AlphabetDataset, BinaryAlphabetDataset
from leakers.datasets.datamodules import GenericAlphabetDatamodule

from leakers.nn.modules.base import LeakerModule
from leakers.trainers.utils import Masquerade
from leakers.utils import AugmentedReality2DUtils, TransformsUtils
import torch.nn.functional as F


class RunesDetector(object):
    def __init__(
        self, model: LeakerModule, dataset: AlphabetDataset, grayscale: bool = False
    ):

        self.model = model
        self.masquerade = Masquerade()
        self.dataset = dataset
        self.th_block_size = 11
        self.th_C = 2
        self.min_max_area_size = [200, 600 * 400]
        self.eps_curve = 0.01
        self.leaker_size = model.image_shape()[-1]
        self._grayscale = grayscale

    def rectangle_info(self, points):
        ## TODO: TEMP DEBUG To detect Rectangle ratio?

        c0 = points[0, :]
        c1 = points[1, :]
        c2 = points[2, :]
        c3 = points[3, :]

        d0 = np.linalg.norm(c0 - c1)
        d1 = np.linalg.norm(c1 - c2)
        d2 = np.linalg.norm(c2 - c3)
        d3 = np.linalg.norm(c3 - c0)
        ratios = [d0 / d1, d0 / d2, d0 / d3]

    def detect_single_leaker(self, leaker_image: np.ndarray):
        """
        Detects a single leaker in the image
        :param leake_image:
        :return:
        """

        if self._grayscale:
            leaker_image = cv2.cvtColor(leaker_image, cv2.COLOR_RGB2GRAY)

        if leaker_image.dtype == np.uint8:
            leaker_image = leaker_image / 255.0

        if self._grayscale:
            x = torch.Tensor(leaker_image).unsqueeze(0).unsqueeze(0)
        else:
            x = torch.Tensor(leaker_image).unsqueeze(0).permute(0, 3, 1, 2)

        x = F.interpolate(
            x,
            size=[self.leaker_size, self.leaker_size],
            mode="bilinear",
            align_corners=False,
        )
        # predict code / word idx
        out = self.model.encode(x)
        code = out["code"]
        code_idx = self.dataset.words_to_indices(code.detach().cpu().numpy())

        return {"code": code_idx[0], "rot": 0}

    # def detect_multi_leaker(self, leaker_image: np.ndarray, splits: int = 2):

    #     h, w = leaker_image.shape[:2]
    #     sh, sw = h // splits, w // splits

    def generate_raw_leakers(
        self, batch_size: int = 1
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Generate raw leakers as  ID,Image pairs.

        :param batch_size: batch size input for image generation, defaults to 1
        :type batch_size: int, optional
        :return: list of dict [{"id": id, "image": image}]
        :rtype: Sequence[Dict[str, np.ndarray]]
        """

        datamodule = GenericAlphabetDatamodule(
            dataset=self.dataset, batch_size=batch_size, drop_last=False
        )
        leakers = []
        for sample in datamodule.val_dataloader():

            imgs = self.model.generate(sample["x"])
            imgs = self.masquerade(imgs)
            y = sample["y"]

            B, C, H, W = imgs.shape
            for b in range(B):
                leakers.append(
                    {
                        "id": y[b].item(),
                        "image": imgs[b, ::].permute(1, 2, 0).detach().cpu().numpy(),
                    }
                )

        return leakers

    def generate_leakers(
        self,
        output_size: int = 128,
        border: int = 1,
        padding: int = 16,
        batch_size: int = 1,
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Generate Leakers adding padding/border to raw leakers.

        :param output_size: desired output size, defaults to 128
        :type output_size: int, optional
        :param border: black border width, defaults to 1
        :type border: int, optional
        :param padding: white padding width, defaults to 16
        :type padding: int, optional
        :param batch_size: ibatch size input for image generation, defaults to 1
        :type batch_size: int, optional
        :return: list of dict [{"id": id, "image": image}]
        :rtype: Sequence[Dict[str, np.ndarray]]
        """

        PAD = lambda x, padding, color: cv2.copyMakeBorder(
            x,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[color, color, color],
        )

        raw_leakers = self.generate_raw_leakers(batch_size=batch_size)
        for raw_leaker in raw_leakers:
            raw_leaker["image"] = PAD(PAD(raw_leaker["image"], border, 0), padding, 1)
            raw_leaker["image"] = cv2.resize(
                raw_leaker["image"],
                (output_size, output_size),
                interpolation=cv2.INTER_LINEAR,
            )

        return raw_leakers
