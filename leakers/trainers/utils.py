from typing import Dict, Sequence, Tuple
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_lightning.loggers.base import LoggerCollection
from torchvision.utils import make_grid


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
