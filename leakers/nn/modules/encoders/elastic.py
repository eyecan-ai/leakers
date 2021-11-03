from typing import Optional, Sequence, Tuple
from warnings import filters
import torch
import numpy as np
from torch.nn.modules.linear import Identity


class ElasticEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_size: int = 32,
        n_layers: int = 4,
        cin: int = 32,
        k: int = 3,
        act_middle: Optional[str] = "torch.nn.ReLU",
        act_last: Optional[str] = "torch.nn.Tanh",
        bn: bool = False,
    ):
        super().__init__()

        self._input_shape = np.array(input_shape)
        self._input_channels = self._input_shape[0]

        # Computes final features map shape
        fh, fw = (self._input_shape[1:3] / (2 ** n_layers)).astype(np.int32)
        assert np.all(np.array([fh, fw]) >= 2), "Number of layers to big."

        # Creates a Sequential modules list
        layers = []

        # number of per-layer-filters
        filters = [self._input_channels] + [cin * (2 ** i) for i in range(n_layers)]
        for i in range(n_layers):
            layers.append(
                self._build_basic_block(
                    filters[i],
                    filters[i + 1],
                    first_bn=(i == 0),
                    k=k,
                    bn=bn,
                    act=act_middle,
                )
            )

        # Flattening features and Linear layer
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(fh * fw * filters[-1], latent_size))
        layers.append(eval(act_last)() if act_last is not None else Identity())

        # Filters Identities
        layers = [x for x in layers if not isinstance(x, Identity)]
        self.layers = torch.nn.Sequential(*layers)

    def _build_basic_block(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        bn: bool = True,
        first_bn: bool = False,
        act: str = "torch.nn.ReLU",
    ):
        pad = int((k - 1) / 2)
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, k, 1, pad),
            torch.nn.BatchNorm2d(cout) if not first_bn and bn else Identity(),
            eval(act)(),
            torch.nn.Conv2d(cout, cout, k, 2, pad),
            torch.nn.BatchNorm2d(cout) if bn else Identity(),
            eval(act)(),
        )
        layers = [x for x in layers if not isinstance(x, Identity)]
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnFlatten(torch.nn.Module):
    def __init__(self, shape: Sequence[int]):
        super(UnFlatten, self).__init__()
        self.shape = shape
        assert len(self.shape) == 3, "UnFlatten: needs shape like [C,H,W]"

    def forward(self, x):
        return x.view(x.size(0), self.shape[0], self.shape[1], self.shape[2])


class ElasticDecoder(torch.nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        latent_size: int = 32,
        n_layers: int = 4,
        cin: int = 256,
        k: int = 3,
        act_middle: str = "torch.nn.ReLU",
        act_last: str = "torch.nn.Sigmoid",
        bn: bool = False,
    ):

        super().__init__()

        self._output_shape = np.array(output_shape)
        self._output_channels = output_shape[0]

        fh, fw = (self._output_shape[1:3] / (2 ** n_layers)).astype(np.int32)
        assert np.all(np.array([fh, fw]) >= 2), "number of layers to big."

        # Creates a Sequential modules list
        layers = []

        # Initial Linear layer on reshaped features
        initial_size = cin * fh * fw

        layers.append(torch.nn.Linear(latent_size, initial_size))
        layers.append(UnFlatten([cin, fh, fw]))

        filters = [cin // (2 ** i) for i in range(n_layers + 1)] + [
            self._output_channels
        ]

        for i in range(n_layers):
            layers.append(
                self._build_basic_block(
                    cin=filters[i], cout=filters[i + 1], k=k, bn=bn, act=act_middle
                )
            )

        layers.append(
            self._build_last_block(
                cin=filters[-2], cout=filters[-1], k=k, bn=bn, act=act_middle
            )
        )
        layers.append(eval(act_last)() if act_last is not None else Identity())
        self.layers = torch.nn.Sequential(*layers)

    def _build_basic_block(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        bn: bool = True,
        first_bn: bool = False,
        act: str = "torch.nn.ReLU",
    ):

        padding = int((k - 1) / 2)
        layers = [
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.BatchNorm2d(cin) if not first_bn and bn else Identity(),
            eval(act)(),
            torch.nn.Conv2d(
                cin,
                cout,
                kernel_size=k,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(cout) if bn else Identity(),
            eval(act)(),
        ]
        layers = [x for x in layers if not isinstance(x, Identity)]
        return torch.nn.Sequential(*layers)

    # def _build_last_block(self, cin: int, cout: int, k: int = 3):
    #     pad = int((k - 1) / 2)
    #     return torch.nn.Conv2d(
    #         cin,
    #         cout,
    #         kernel_size=k,
    #         stride=1,
    #         padding=pad,
    #     )

    def _build_last_block(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        bn: bool = True,
        act: str = "torch.nn.ReLU",
    ):
        pad = int((k - 1) / 2)
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, k, 1, pad),
            eval(act)(),
            torch.nn.Conv2d(cout, cout, k, 1, pad),
        )
        layers = [x for x in layers if not isinstance(x, Identity)]
        return torch.nn.Sequential(*layers)

    def forward(self, e):
        return self.layers(e)
