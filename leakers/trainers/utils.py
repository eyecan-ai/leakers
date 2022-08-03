from random import random
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_lightning.loggers.base import LoggerCollection
from torchvision.utils import make_grid
import torch.nn.functional as F


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

    def _tensorboard_log_image(
        self, exp: SummaryWriter, name: str, x: torch.Tensor, step: int
    ) -> None:
        x = (x.detach().cpu() * 255).type(torch.uint8)
        exp.add_image(name, x, step)


class TensorUtils:
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


class Masquerade(torch.nn.Module):
    def __init__(
        self,
        size: int = 128,
        mask_type: str = "circle",
        mask_ratio: float = 0.8,
        mask_background: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()

        self._size = size
        self._mask_type = mask_type
        self._mask_ratio = mask_ratio
        self._mask_background = mask_background
        self._randomize_mask = mask_background is None

        TT = lambda x: torch.Tensor(x / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        if mask_type == "circle":
            radius = int(size * mask_ratio / 2)
            self.mask = np.zeros((self._size, self._size, 3), dtype=np.uint8)
            self.mask = cv2.circle(
                self.mask,
                (self._size // 2, self._size // 2),
                radius,
                (255, 255, 255),
                -1,
            )
        else:
            raise NotImplementedError(f"Mask type [{self._mask_type}] not implemented")

        self.mask = TT(self.mask)

        if self._randomize_mask:
            self.mask_background = (
                np.ones((self._size, self._size, 3), dtype=np.uint8) * 255
            )
            self.mask_background = TT(self.mask_background)
        else:
            self.mask_background = (
                torch.Tensor(np.array(self._mask_background) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .permute(1, 0, 2, 3)
                .repeat(1, 1, self._size, self._size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        mask = F.interpolate(
            self.mask,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask = mask.repeat(B, 1, 1, 1)

        mask_background = F.interpolate(
            self.mask_background,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        mask_background = mask_background.repeat(B, 1, 1, 1)

        if self._randomize_mask:
            random_color = torch.rand(B, 3, 1, 1)
            mask_background *= random_color

        mask = mask.to(x.device)
        mask_background = mask_background.to(x.device)
        x = x * mask + mask_background * (1 - mask)
        return x


class MasqueradeByImage(torch.nn.Module):
    def __init__(
        self,
        image_filename: str,
        size: int = 128,
        mask_background: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()

        import imageio

        self._size = size
        self._image_filename = image_filename
        self._image = imageio.imread(image_filename)[:, :, :3]
        self._mask_background = mask_background
        self._randomize_mask = mask_background is None

        TT = lambda x: torch.Tensor(x / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        self.mask = self._image
        if len(self.mask.shape) == 2:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)

        self.mask = TT(self.mask)

        if self._randomize_mask:
            self.mask_background = (
                np.ones((self._size, self._size, 3), dtype=np.uint8) * 255
            )
            self.mask_background = TT(self.mask_background)
        else:
            self.mask_background = (
                torch.Tensor(np.array(self._mask_background) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .permute(1, 0, 2, 3)
                .repeat(1, 1, self._size, self._size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        mask = F.interpolate(
            self.mask,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask = mask.repeat(B, 1, 1, 1)

        mask_background = F.interpolate(
            self.mask_background,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        mask_background = mask_background.repeat(B, 1, 1, 1)

        if self._randomize_mask:
            random_color = torch.rand(B, 3, 1, 1)
            mask_background *= random_color

        mask = mask.to(x.device)
        mask_background = mask_background.to(x.device)
        x = x * mask + mask_background * (1 - mask)
        return x


class MasqueradeRandom(torch.nn.Module):
    def __init__(
        self,
        size: int = 32,
        mask_type: List[str] = ["none", "circle"],
        mask_ratio: float = -1,
    ) -> None:
        super().__init__()

        self._size = size
        self._mask_type = mask_type
        self._mask_ratio = mask_ratio

    def _totensor(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor(x / 255.0).permute(2, 0, 1).float().unsqueeze(0)

    def _generate_mask(self) -> torch.Tensor:

        mask_type = random.choice(self._mask_type)

        if mask_type == "none":
            mask = np.ones((self._size, self._size, 3), dtype=np.uint8) * 255
            mask_background = np.ones((self._size, self._size, 3), dtype=np.uint8) * 255
        elif mask_type == "circle":
            if self._mask_ratio > 0:
                radius = int(self._size * self._mask_ratio / 2)
            else:
                ratio = np.random.uniform(0.5, 1.0)
                radius = int(self._size * ratio / 2)

            mask = np.zeros((self._size, self._size, 3), dtype=np.uint8)
            mask_background = np.ones((self._size, self._size, 3), dtype=np.uint8) * 255
            mask = cv2.circle(
                mask,
                (self._size // 2, self._size // 2),
                radius,
                (255, 255, 255),
                -1,
            )
        else:
            raise NotImplementedError(f"Mask type [{self._mask_type}] not implemented")

        # convert
        mask = self._totensor(mask)
        mask_background = self._totensor(mask_background)

        return mask, mask_background

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        mask, mask_background = self._generate_mask()

        mask = F.interpolate(
            mask,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask = mask.repeat(B, 1, 1, 1)

        mask_background = F.interpolate(
            mask_background,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask_background = mask_background.repeat(B, 1, 1, 1)

        mask = mask.to(x.device)
        mask_background = mask_background.to(x.device)
        x = x * mask + mask_background * (1 - mask)
        return x
