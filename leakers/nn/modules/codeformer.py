import os
import copy
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from leakers.nn.modules.base import LeakerModule
from leakers.nn.modules.elastic import ElasticDecoder

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImageNetNormalizer(torch.nn.Module):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        super().__init__()
        assert len(mean) == len(std)
        self._C = len(mean)
        self._mean = (
            torch.tensor(mean, requires_grad=False).float().view(1, self._C, 1, 1)
        )
        self._std = (
            torch.tensor(std, requires_grad=False).float().view(1, self._C, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        mean = self._mean.to(x.device).repeat(B, 1, H, W)
        std = self._std.to(x.device).repeat(B, 1, H, W)
        out = (x - mean) / std
        return out


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=16,
        stride=16,
        padding=0,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: type = nn.GELU,
        norm_layer: type = GroupNorm,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(
    dim,
    index,
    layers,
    pool_size=3,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    norm_layer=GroupNorm,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            PoolFormerBlock(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class EyePoolFormer(nn.Module):
    """PoolFormer class
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims,-mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    """

    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        pool_size=3,
        norm_layer=GroupNorm,
        act_layer=nn.GELU,
        code_size=1000,
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        extract_features=False,
        init_cfg=None,
        pretrained=None,
        act_last=None,
        **kwargs,
    ):

        super().__init__()

        if not extract_features:
            self.num_classes = code_size
        self.fork_feat = extract_features

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0],
        )

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)

        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Sequential(
            nn.Linear(embed_dims[-1], code_size),
            eval(act_last)() if act_last is not None else nn.Identity(),
        )

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if self.init_cfg is None and pretrained is None:
            print(
                f"No pre-trained weights for "
                f"{self.__class__.__name__}, "
                f"training start from scratch"
            )
            pass
        else:
            assert "checkpoint" in self.init_cfg, (
                f"Only support "
                f"specify `Pretrained` in "
                f"`init_cfg` in "
                f"{self.__class__.__name__} "
            )
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


class PoolFormerCoder(LeakerModule):
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

        self.encoder = PoolFormerFactory.create(
            "poolformer_s12",
            pretrained=False,
            code_size=code_size + 4,
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


class PoolFormerFactory:
    PRETRAINED_WEIGHTS_PATHS = {
        "poolformer_s12": "pretrained_weights/poolformer_s12.pth.tar",
        "poolformer_s24": "pretrained_weights/poolformer_s24.pth.tar",
        "poolformer_s36": "pretrained_weights/poolformer_s36.pth.tar",
        "poolformer_m36": "pretrained_weights/poolformer_m36.pth.tar",
        "poolformer_m48": "pretrained_weights/poolformer_m48.pth.tar",
    }

    def BUILD_MODEL_CFG(url="", **kwargs):
        return {
            "url": url,
            "num_classes": 1000,
            "input_size": (3, 224, 224),
            "pool_size": None,
            "crop_pct": 0.95,
            "interpolation": "bicubic",
            "mean": IMAGENET_DEFAULT_MEAN,
            "std": IMAGENET_DEFAULT_STD,
            "classifier": "head",
            **kwargs,
        }

    DEFAULT_CFGS = {
        "poolformer_s": BUILD_MODEL_CFG(crop_pct=0.9),
        "poolformer_m": BUILD_MODEL_CFG(crop_pct=0.95),
    }

    MODEL_TYPE = {
        "poolformer_s12": {
            "layers": [2, 2, 6, 2],
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "cfg_type": "poolformer_s",
        },
        "poolformer_s24": {
            "layers": [4, 4, 12, 4],
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "cfg_type": "poolformer_s",
        },
        "poolformer_s36": {
            "layers": [6, 6, 18, 6],
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "cfg_type": "poolformer_s",
        },
        "poolformer_m36": {
            "layers": [6, 6, 18, 6],
            "embed_dims": [96, 192, 384, 768],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "cfg_type": "poolformer_m",
        },
        "poolformer_m48": {
            "layers": [8, 8, 24, 8],
            "embed_dims": [96, 192, 384, 768],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "cfg_type": "poolformer_m",
        },
    }

    @staticmethod
    def create(name, pretrained=True, **kwargs):

        params = PoolFormerFactory.MODEL_TYPE[name]

        # Create PoolFormer
        model = EyePoolFormer(
            layers=params["layers"],
            embed_dims=params["embed_dims"],
            mlp_ratios=params["mlp_ratios"],
            downsamples=params["downsamples"],
            **kwargs,
        )
        # model.default_cfg = default_cfgs[params["cfg_type"]]

        if pretrained:
            model = PoolFormerFactory.load_pretrained_weights(
                model,
                PoolFormerFactory.PRETRAINED_WEIGHTS_PATHS[name],
            )
        return model

    def load_pretrained_weights(model, filename):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(filename, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
