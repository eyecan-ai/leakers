import math
from typing import Any, Dict, Sequence, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UnFlatten(nn.Module):
    def __init__(self, shape: Sequence[int]):
        super(UnFlatten, self).__init__()
        self.shape = shape
        assert len(self.shape) == 3, "UnFlatten: needs shape like [C,H,W]"

    def forward(self, x):
        return x.view(x.size(0), self.shape[0], self.shape[1], self.shape[2])


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: int = 3,
        with_batch_norm: bool = False,
    ):
        """Simple Conv->BN->RELU->Conv block

        :param in_filters: previous layer filters size
        :type in_filters: int
        :param out_filters: output filters size
        :type out_filters: int
        :param kernel_size: kernel size (must be odd), defaults to 3
        :type kernel_size: int, optional
        :param with_batch_norm: TRUE to use batchnorm, defaults to False
        :type with_batch_norm: bool, optional
        """
        super(BasicBlock, self).__init__()

        assert kernel_size % 2 != 0, "BasicBlock: kernel_size must be odd."

        padding = int((kernel_size - 1) / 2)
        layers = [
            nn.Conv2d(
                in_filters,
                in_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        ]
        if with_batch_norm:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv2d(
                in_filters,
                out_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CommonBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        first_block: bool = False,
        kernel_size: int = 3,
        stride: int = 2,
        with_batch_norm: bool = True,
    ):
        """Common downsample block Conv->BN->RELU-Conv->BN->Relu where the second Convolutional block downsamples the input

        :param in_filters: previous layer filters size
        :type in_filters: int
        :param out_filters: output filters size
        :type out_filters: int
        :param first_block: TRUE to deactivate input batch normalization, defaults to False
        :type first_block: bool, optional
        :param kernel_size: kernel size (must be odd), defaults to 3
        :type kernel_size: int, optional
        :param stride: factor (must be a power of 2), defaults to 2
        :type stride: int, optional
        :param with_batch_norm: TRUE to activate batch normalization, defaults to True
        :type with_batch_norm: bool, optional
        """
        super(CommonBlock, self).__init__()

        assert kernel_size % 2 != 0, "CommonBlock: kernel_size must be odd."
        assert math.log(
            stride, 2
        ).is_integer(), "CommonBlock: stride must be a power of 2."

        padding = int((kernel_size - 1) / 2)
        layers = [
            nn.Conv2d(
                in_filters,
                out_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        ]
        if not first_block and with_batch_norm:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv2d(
                out_filters,
                out_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        if with_batch_norm:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CommonBlockUp(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        first_block: bool = False,
        use_deconvolutions: bool = False,
        kernel_size: int = 3,
        stride: int = 2,
        with_batch_norm: bool = True,
    ):
        """Common upsample block Conv->BN->RELU-Conv->BN->Relu where the first Convolutional block upsamples the input image.

        :param in_filters: previous layer filters size
        :type in_filters: int
        :param out_filters: output filters size
        :type out_filters: int
        :param first_block: TRUE to deactivate input batch normalization, defaults to False
        :type first_block: bool, optional
        :param use_deconvolutions: FALSE to use simple Upsample, defaults to True
        :type use_deconvolutions: bool, optional
        :param kernel_size: kernel size (must be odd), defaults to 3
        :type kernel_size: int, optional
        :param stride: factor (must be a power of 2), defaults to 2
        :type stride: int, optional
        :param with_batch_norm: TRUE to activate batch normalization, defaults to True
        :type with_batch_norm: bool, optional
        """
        super(CommonBlockUp, self).__init__()

        assert kernel_size % 2 != 0, "CommonBlockUp: kernel_size must be odd."
        assert math.log(
            stride, 2
        ).is_integer(), "CommonBlockUp: stride must be a power of 2."

        padding = int((kernel_size - 1) / 2)
        layers = []
        if not use_deconvolutions:
            layers.append(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True)
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_filters,
                    in_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=padding,
                )
            )
            if not first_block and with_batch_norm:
                layers.append(nn.BatchNorm2d(in_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv2d(
                in_filters,
                out_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        if with_batch_norm:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BasicEncoder(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        input_channels: int,
        latent_size: int = 1000,
        n_layers: int = 4,
        initial_filters: int = 32,
        layers_mult: int = 2,
        kernel_size: int = 3,
        last_activation: Optional[torch.nn.Module] = nn.Tanh(),
        with_batch_norm: bool = False,
    ):
        """[B,C,H,W] -> BasicEncoder -> [B, latent_size]

        :param input_size: [description]
        :type input_size: Tuple[int, int]
        :param input_channels: [description]
        :type input_channels: int
        :param latent_size: [description], defaults to 1000
        :type latent_size: int, optional
        :param n_layers: [description], defaults to 4
        :type n_layers: int, optional
        :param initial_filters: [description], defaults to 32
        :type initial_filters: int, optional
        :param layers_mult: [description], defaults to 2
        :type layers_mult: int, optional
        :param kernel_size: [description], defaults to 3
        :type kernel_size: int, optional
        :param last_activation: [description], defaults to nn.Tanh()
        :type last_activation: Optional[torch.nn.Module], optional
        :param with_batch_norm: [description], defaults to False
        :type with_batch_norm: bool, optional
        """

        super().__init__()

        assert kernel_size % 2 != 0, "GenericEncoder: kernel_size must be odd."
        assert math.log(
            initial_filters, 2
        ).is_integer(), "GenericEncoder: stride must be a power of 2."

        self.input_size = np.array(input_size)
        self.input_channels = input_channels

        # Computes final size before features layer
        final_size = (self.input_size / (2 ** n_layers)).astype(np.int32)

        assert np.all(final_size >= 2), "GenericEncoder: number of layers to big."

        # Creates a Sequential modules list
        self.layers = nn.Sequential()

        start_filters = input_channels
        last_filters = initial_filters
        for i in range(n_layers):
            self.layers.add_module(
                "layer_{}".format(i),
                CommonBlock(
                    start_filters,
                    last_filters,
                    first_block=(i == 0),
                    kernel_size=kernel_size,
                    with_batch_norm=with_batch_norm,
                ),
            )
            start_filters = last_filters
            last_filters = start_filters * layers_mult

        # Flattening features and Linear layer
        self.layers.add_module("flatten", nn.Flatten())
        self.layers.add_module(
            "features",
            nn.Linear(final_size[0] * final_size[1] * start_filters, latent_size),
        )

        # Last activation if any
        if last_activation is not None:
            self.layers.add_module("activation", last_activation)

    def forward(self, x):
        return self.layers(x)


class BasicDecoder(nn.Module):
    def __init__(
        self,
        output_size,
        output_channels,
        latent_size: int = 1000,
        n_layers: int = 4,
        initial_filters: int = 256,
        layers_mult: int = 2,
        kernel_size: int = 3,
        last_activation: torch.nn.Module = nn.Sigmoid(),
        use_deconvolutions: bool = False,
        with_batch_norm: bool = False,
    ):
        """[B, latent_size] -> BasicDecoder -> [B,C,H,W]

        :param output_size: [description]
        :type output_size: [type]
        :param output_channels: [description]
        :type output_channels: [type]
        :param latent_size: [description], defaults to 1000
        :type latent_size: int, optional
        :param n_layers: [description], defaults to 4
        :type n_layers: int, optional
        :param initial_filters: [description], defaults to 256
        :type initial_filters: int, optional
        :param layers_mult: [description], defaults to 2
        :type layers_mult: int, optional
        :param kernel_size: [description], defaults to 3
        :type kernel_size: int, optional
        :param last_activation: [description], defaults to nn.Sigmoid()
        :type last_activation: torch.nn.Module, optional
        :param use_deconvolutions: [description], defaults to False
        :type use_deconvolutions: bool, optional
        :param with_batch_norm: [description], defaults to False
        :type with_batch_norm: bool, optional
        """

        super().__init__()

        assert kernel_size % 2 != 0, "GenericDecoder: kernel_size must be odd."
        assert math.log(
            initial_filters, 2
        ).is_integer(), "GenericDecoder: stride must be a power of 2."

        self.output_size = np.array(output_size)
        final_size = (self.output_size / (2 ** n_layers)).astype(np.int32)
        assert np.all(final_size >= 2), "GenericDecoder: number of layers to big."

        # Creates a Sequential modules list
        self.layers = nn.Sequential()

        # Initial Linear layer on reshaped features
        initial_size = initial_filters * final_size[0] * final_size[1]
        self.layers.add_module("Features", nn.Linear(latent_size, initial_size))
        self.layers.add_module(
            "UnFlatten", UnFlatten([initial_filters, final_size[0], final_size[1]])
        )

        start_filters = initial_filters
        last_filters = int(initial_filters / layers_mult)
        for i in range(n_layers):
            self.layers.add_module(
                "layer_{}".format(i),
                CommonBlockUp(
                    start_filters,
                    last_filters,
                    use_deconvolutions=use_deconvolutions,
                    kernel_size=kernel_size,
                    with_batch_norm=with_batch_norm,
                ),
            )
            # self.layers.add_module("upsample_{}".format(i), nn.Upsample(scale_factor=2))

            start_filters = last_filters
            last_filters = int(start_filters / layers_mult)

        self.layers.add_module(
            "last_layer",
            BasicBlock(
                start_filters,
                output_channels,
                kernel_size=kernel_size,
                with_batch_norm=False,
            ),
        )

        if last_activation is not None:
            self.layers.add_module("activation", last_activation)

    def forward(self, e):
        return self.layers(e)
