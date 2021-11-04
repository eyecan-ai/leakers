from abc import abstractmethod
from typing import Optional, Tuple
import torch
import kornia.geometry.transform as KT


class LeakerModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def _rotate(
        self, image: torch.Tensor, k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Rotate image batch with rotation classes [0= 0°; 1=90°; 2=180°; 3=270°]

        :param image: input images batch [B,C,H,W]
        :type image: torch.Tensor
        :param k: input angle classes [B]
        :type k: torch.Tensor
        :return: rotatate images batch [B,C,H,W]
        :rtype: torch.Tensor
        """

        if k is None:
            k = torch.Tensor([0]).repeat(image.shape[0]).to(image.device)

        return KT.rotate(
            image, 90.0 * k
        )  # TODO: replace with torch deafult rot90 per batch?

    @abstractmethod
    def generate(self, code: torch.Tensor, angle_classes: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
