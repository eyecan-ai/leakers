import torch
import kornia.augmentation as K
import numpy as np
from torch.nn.modules.linear import Identity


class VirtualRandomizer(torch.nn.Module):
    EPS = 1e-10

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # Image shape
        image_shape = np.array(kwargs.get("image_shape", [3, 64, 64]))
        image_size = image_shape[1:3]

        # Color Jitter
        color_jitter = kwargs.get("color_jitter", True)
        color_jitter_brightness = kwargs.get("color_jitter_brightness", 0.5)
        color_jitter_contrast = kwargs.get("color_jitter_contrast", 0.5)
        color_jitter_saturation = kwargs.get("color_jitter_saturation", 0.5)
        color_jitter_p = kwargs.get("color_jitter_p", 0.5)

        # Random Crop
        random_crop_percentage = kwargs.get("random_crop_percentage", -1)
        random_crop = random_crop_percentage > 0 and random_crop_percentage <= 1.0
        random_crop_size = image_size * random_crop_percentage

        # Channel Shuffle
        channel_shuffle = kwargs.get("channel_shuffle", False)
        channel_shuffle_p = kwargs.get("channel_shuffle_p", 0.5)

        # Gaussian Noise
        gaussian_noise = kwargs.get("gaussian_noise", True)
        gaussian_noise_mean = kwargs.get("gaussian_noise_mean", 0.0)
        gaussian_noise_std = kwargs.get("gaussian_noise_std", 0.05)
        gaussian_noise_p = kwargs.get("gaussian_noise_p", 0.5)

        # Random Affine
        random_affine = kwargs.get("random_affine", True)
        random_affine_rotation = kwargs.get("random_affine_rotation", 15)
        random_affine_translate = kwargs.get("random_affine_translate", [0.1, 0.1])
        random_affine_scale = kwargs.get("random_affine_scale", [0.9, 1.1])
        random_affine_p = kwargs.get("random_affine_p", 0.5)

        # Box Blur
        box_blur = kwargs.get("box_blur", True)
        box_blur_size = kwargs.get("box_blur_size", [35, 35])
        box_blur_p = kwargs.get("box_blur_p", 0.5)

        # Random Erasing
        random_erasing = kwargs.get("random_erasing", True)
        random_erasing_scale = kwargs.get("random_erasing_scale", [0.02, 0.1])
        random_erasing_ratio = kwargs.get("random_erasing_ratio", [0.8, 1.2])
        random_erasing_p = kwargs.get("random_erasing_p", 0.5)

        layers = torch.nn.Sequential(
            # ----------------------------
            # Color Jitter
            K.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                p=color_jitter_p,
            )
            if color_jitter
            else Identity(),
            # ----------------------------
            # Random Crop
            K.RandomCrop(random_crop_size, p=1.0) if random_crop else Identity(),
            torch.nn.Upsample(image_size) if random_crop else Identity(),
            # ----------------------------
            # Channel Shuffe
            K.RandomChannelShuffle(p=channel_shuffle_p)
            if channel_shuffle
            else Identity(),
            # ----------------------------
            # Gaussian Noise
            K.RandomGaussianNoise(
                mean=gaussian_noise_mean, std=gaussian_noise_std, p=gaussian_noise_p
            )
            if gaussian_noise
            else Identity(),
            # ----------------------------
            # Random Affine
            K.RandomAffine(
                random_affine_rotation,
                translate=torch.Tensor(random_affine_translate),
                scale=torch.Tensor(random_affine_scale),
                p=random_affine_p,
            )
            if random_affine
            else Identity(),
            # ----------------------------
            # Box Blurr
            K.RandomBoxBlur(kernel_size=box_blur_size, p=box_blur_p)
            if box_blur
            else Identity(),
            # ----------------------------
            # Random Erasing
            K.RandomErasing(
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                p=random_erasing_p,
            )
            if random_erasing
            else Identity(),
        )

        layers = [x for x in layers if not isinstance(x, Identity)]
        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0 + self.EPS, 1.0 - self.EPS)  # Nan if overflow!
        return self._layers(x)
