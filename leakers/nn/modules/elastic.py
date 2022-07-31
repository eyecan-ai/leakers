from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np
from torch.nn.modules.linear import Identity
from einops.layers.torch import Rearrange
from leakers.nn.modules.base import LeakerModule


def elastic_initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(
            m.weight.data, gain=torch.nn.init.calculate_gain("relu")
        )
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


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
        fh, fw = (self._input_shape[1:3] / (2**n_layers)).astype(np.int32)
        assert np.all(np.array([fh, fw]) >= 2), "Number of layers to big."

        # Creates a Sequential modules list
        layers = []

        # number of per-layer-filters
        filters = [self._input_channels] + [cin * (2**i) for i in range(n_layers)]
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

        self.apply(elastic_initialize_weights)

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


class ElasticDecoder(torch.nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        latent_size: int = 32,
        n_layers: int = 4,
        cin: int = 32,
        k: int = 3,
        act_middle: str = "torch.nn.ReLU",
        act_last: str = "torch.nn.Sigmoid",
        bn: bool = False,
    ):

        super().__init__()

        self._output_shape = np.array(output_shape)
        self._output_channels = output_shape[0]

        fh, fw = (self._output_shape[1:3] / (2**n_layers)).astype(np.int32)
        assert np.all(np.array([fh, fw]) >= 2), "number of layers to big."

        # Creates a Sequential modules list
        layers = []

        # Initial Linear layer on reshaped features
        cout = cin * (2 ** (n_layers - 1))
        initial_size = cout * fh * fw

        layers.append(torch.nn.Linear(latent_size, initial_size))
        layers.append(Rearrange("b (c h w) -> b c h w", c=cout, h=fh, w=fw))

        filters = [cout // (2**i) for i in range(n_layers + 1)] + [
            self._output_channels
        ]

        for i in range(n_layers):
            layers.append(
                self._build_basic_block(
                    cin=filters[i],
                    cout=filters[i + 1],
                    k=k,
                    bn=bn,
                    act=act_middle,
                )
            )

        layers.append(
            self._build_last_block(
                cin=filters[-2],
                cout=filters[-1],
                k=k,
                bn=bn,
            )
        )
        layers.append(eval(act_last)() if act_last is not None else Identity())
        self.layers = torch.nn.Sequential(*layers)

        self.apply(elastic_initialize_weights)

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
            torch.nn.Conv2d(
                cin,
                cout,
                kernel_size=k,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(cout) if bn else Identity(),
            eval(act)(),
            torch.nn.Conv2d(
                cout,
                cout,
                kernel_size=k,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(cout) if not first_bn and bn else Identity(),
            eval(act)(),
        ]
        layers = [x for x in layers if not isinstance(x, Identity)]
        return torch.nn.Sequential(*layers)

    def _build_last_block(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        bn: bool = True,
    ):
        pad = int((k - 1) / 2)
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, k, 1, pad),
            torch.nn.ReLU(),
            torch.nn.Conv2d(cout, cout, k, 1, pad),
        )
        layers = [x for x in layers if not isinstance(x, Identity)]
        return torch.nn.Sequential(*layers)

    def forward(self, e):
        return self.layers(e)


class ElasticCoder(LeakerModule):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        code_size: int = 32,
        cin: int = 32,
        n_layers: int = 4,
        k: int = 3,
        bn: bool = False,
        act_middle: str = "torch.nn.ReLU",
        act_latent: str = None,  # "torch.nn.Sigmoid",
        act_last: str = "torch.nn.Sigmoid",
    ) -> None:
        super().__init__()

        self._code_size = code_size
        self._image_shape = image_shape

        self.encoder = ElasticEncoder(
            input_shape=image_shape,
            latent_size=code_size + 4,
            n_layers=n_layers,
            cin=cin,
            k=k,
            act_middle=act_middle,
            act_last=act_latent,
            bn=bn,
        )

        self.decoder = ElasticDecoder(
            output_shape=image_shape,
            latent_size=code_size,
            n_layers=n_layers,
            cin=cin,
            act_middle=act_middle,
            act_last=act_last,
            bn=bn,
        )

    def code_size(self) -> int:
        return self._code_size

    def image_shape(self) -> Tuple[int, int, int]:
        return self._image_shape

    def generate(
        self, code: torch.Tensor, angle_classes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        image = self.decoder(code)
        image_rot = self._rotate(image, k=angle_classes)
        return image_rot

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(images)
        code = z[:, : self._code_size]
        scores = torch.abs(code)
        rot_logits = z[:, self._code_size : self._code_size + 4]
        rot_classes = torch.argmax(rot_logits, dim=1)
        return {
            "code": code,
            "scores": scores,
            "rot_logits": rot_logits,
            "rot_classes": rot_classes,
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = self.generate(x)
        code = self.encode(gen)["code"]
        return code


from pydantic import BaseModel, PrivateAttr, Extra


class CoderElastic(LeakerModule):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        word_size: int = 32,
        input_channels: int = 32,
        layers: int = 4,
        kernel_size: int = 3,
        use_batchnorm: bool = False,
        activation_middle: str = "torch.nn.ReLU",
        activation_latent: Optional[str] = None,  # "torch.nn.Sigmoid",
        activation_output: str = "torch.nn.Sigmoid",
    ) -> None:
        super().__init__()

        self.shape = shape
        self.word_size = word_size
        self.input_channels = input_channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm
        self.activation_middle = activation_middle
        self.activation_latent = activation_latent
        self.activation_output = activation_output

        self._encoder = ElasticEncoder(
            input_shape=self.shape,
            latent_size=self.word_size + 4,
            n_layers=self.layers,
            cin=self.input_channels,
            k=self.kernel_size,
            act_middle=self.activation_middle,
            act_last=self.activation_latent,
            bn=self.use_batchnorm,
        )

        self._decoder = ElasticDecoder(
            output_shape=self.shape,
            latent_size=self.word_size,
            n_layers=self.layers,
            cin=self.input_channels,
            act_middle=self.activation_middle,
            act_last=self.activation_output,
            bn=self.use_batchnorm,
        )

    def dict(self) -> dict:
        return {
            "shape": self.shape,
            "word_size": self.word_size,
            "input_channels": self.input_channels,
            "layers": self.layers,
            "kernel_size": self.kernel_size,
            "use_batchnorm": self.use_batchnorm,
            "activation_middle": self.activation_middle,
            "activation_latent": self.activation_latent,
            "activation_output": self.activation_output,
        }

    def code_size(self) -> int:
        return self.word_size

    def image_shape(self) -> Tuple[int, int, int]:
        return self.shape

    def generate(
        self, code: torch.Tensor, angle_classes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        image = self._decoder(code)
        image_rot = self._rotate(image, k=angle_classes)
        return image_rot

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self._encoder(images)
        code = z[:, : self.code_size()]
        scores = torch.abs(code)
        rot_logits = z[:, self.code_size() : self.code_size() + 4]
        rot_classes = torch.argmax(rot_logits, dim=1)
        return {
            "code": code,
            "scores": scores,
            "rot_logits": rot_logits,
            "rot_classes": rot_classes,
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = self.generate(x)
        code = self.encode(gen)["code"]
        return code
