import numpy as np
import torch
import kornia.geometry as kgeometry
from typing import Tuple
import transforms3d


class WarpingModule(torch.nn.Module):
    def __init__(
        self,
        camera_matrix: np.ndarray,
        virtual_size: float = 0.05,
    ):
        """Initialize the warping module.

        :param camera_matrix: The camera matrix.
        :type camera_matrix: np.ndarray
        :param virtual_size: virtual size of the marker in world meters, defaults to 0.05
        :type virtual_size: float, optional
        """
        super().__init__()

        self._camera_matrix = torch.Tensor(camera_matrix)
        self._size = virtual_size
        self._points = (
            np.array(
                [
                    [1, -1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [-1, -1, 0],
                ]
            )
            * self._size
            * 0.5
        )

    def points_2d(self, transforms: torch.Tensor) -> torch.Tensor:
        """Get the 2D points of the 4 corners of the square when transform is applied.

        :param transforms: The transformation matrix.
        :type transforms: torch.Tensor
        :return: The 2D projected points of the 4 corners of the square.
        :rtype: torch.Tensor
        """
        B, _, _ = transforms.shape
        points = (
            torch.Tensor(self._points)
            .unsqueeze(0)
            .repeat(B, 1, 1)
            .to(transforms.device)
        )
        t_points = kgeometry.transform_points(transforms, points)
        B, N, _ = t_points.shape

        t_points = t_points.view(B * N, 3)
        camera = self._camera_matrix.repeat(t_points.shape[0], 1, 1).to(
            transforms.device
        )
        p_points = kgeometry.camera.project_points(t_points, camera).view(B, N, 2)
        return p_points

    def get_perspective_transform(
        self, transforms: torch.Tensor, source_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the perspective transform from the transformation matrix .

        :param transforms: The transformation matrix.
        :type transforms: torch.Tensor
        :param source_image_size: The size of the source image.
        :type source_image_size: Tuple[int, int]
        :return: The perspective transform.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        W, H = source_image_size
        source_points = np.array(
            [
                [0, 0],
                [W, 0],
                [W, H],
                [0, H],
            ]
        )
        B, _, _ = transforms.shape
        source_points = (
            torch.Tensor(source_points)
            .unsqueeze(0)
            .repeat(B, 1, 1)
            .to(transforms.device)
        )
        target_points = self.points_2d(transforms=transforms)
        M = kgeometry.get_perspective_transform(source_points, target_points)
        M_inv = kgeometry.get_perspective_transform(target_points, source_points)
        return M, M_inv

    def warp_image(
        self,
        source_image: torch.Tensor,
        transforms: torch.Tensor,
        canvas_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Warp an image using the perspective transform.

        :param source_image: The image to warp.
        :type source_image: torch.Tensor
        :param transforms: The transforms to warp the image with.
        :type transforms: torch.Tensor
        :param canvas_size: the size of the warp output
        :type canvas_size: Tuple[int, int]
        :return: output warped images [B, C, H, W]
        :rtype: torch.Tensor
        """
        mH, mW = source_image.shape[2:4]
        H, W = canvas_size
        M, M_inv = self.get_perspective_transform(
            transforms=transforms, source_image_size=[mH, mW]
        )
        warped_img = kgeometry.warp_perspective(
            source_image, M, (H, W), align_corners=True
        )
        return warped_img

    def unwarp_image(
        self,
        canvas_image: torch.Tensor,
        transforms: torch.Tensor,
        square_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Unwarp an image using the inverse of the perspective transform.

        :param canvas_image: The image to unwarp.
        :type canvas_image: torch.Tensor
        :param transforms: The transforms to unwarp the image with.
        :type transforms: torch.Tensor
        :param square_size: the size of the unwarp output
        :type square_size: Tuple[int, int]
        :return: uwarped images [B, C, H, W]
        :rtype: torch.Tensor
        """
        H, W = canvas_image.shape[2:4]
        mH, mW = square_size
        M, M_inv = self.get_perspective_transform(
            transforms=transforms, source_image_size=[mH, mW]
        )
        unwarped_img = kgeometry.warp_perspective(
            canvas_image, M_inv, (mH, mW), align_corners=True
        )
        return unwarped_img

    @classmethod
    def spherical_marker_transform(
        cls, radius: float, azimuth: float, zenith: float, input_degrees: bool = True
    ) -> np.ndarray:
        """Returns a transformation matrix with spherical coordinates input

        :param radius: distance of the marker
        :type radius: float
        :param azimuth: azimuth angle of the camera w.r.t. marker
        :type azimuth: float
        :param zenith: zenith angle of the camera w.r.t.marker
        :type zenith: float
        :param input_degrees: TRUE to use degrees for angles, defaults to True
        :type input_degrees: bool, optional
        :return: transformation matrix [4, 4]
        :rtype: np.ndarray
        """
        if input_degrees:
            azimuth = azimuth * np.pi / 180.0
            zenith = zenith * np.pi / 180.0

        azimuth_transform = np.eye(4)
        azimuth_transform[:3, :3] = transforms3d.euler.euler2mat(0, 0, azimuth)

        zenith_transform = np.eye(4)
        zenith_transform[:3, :3] = transforms3d.euler.euler2mat(0, zenith, 0)

        radius_transform = np.eye(4)
        radius_transform[2, 3] = radius

        T = np.eye(4)
        T = np.dot(T, radius_transform)
        T = np.dot(T, azimuth_transform)
        T = np.dot(T, zenith_transform)
        return T

    @classmethod
    def random_spherical_transforms(
        cls,
        batch_size: int = 1,
        radius_range: Tuple[float, float] = [0.2, 1.0],
        azimuth_range: Tuple[float, float] = [0, 360.0],
        zenith_range: Tuple[float, float] = [0, 89.0],
        input_degrees: bool = True,
    ) -> torch.Tensor:
        """Generate random spherical transforms as [B,4,4]

        :param batch_size: batch size B, defaults to 1
        :type batch_size: int, optional
        :param radius_range: uniform range [meter] for radius random generation, defaults to [0.2, 1.0]
        :type radius_range: Tuple[float, float], optional
        :param azimuth_range: uniform range for azimuth random generation , defaults to [0, 360.0]
        :type azimuth_range: Tuple[float, float], optional
        :param zenith_range: uniform range for zenith random generation, defaults to [0, 89.0]
        :type zenith_range: Tuple[float, float], optional
        :param input_degrees: TRUE to use degrees for angles, defaults to True
        :type input_degrees: bool, optional
        :return: batch of transforms [B,4,4]
        :rtype: torch.Tensor
        """

        radius = torch.FloatTensor(batch_size).uniform_(
            radius_range[0], radius_range[1]
        )
        azimuth = torch.FloatTensor(batch_size).uniform_(
            azimuth_range[0], azimuth_range[1]
        )
        zenith = torch.FloatTensor(batch_size).uniform_(
            zenith_range[0], zenith_range[1]
        )

        T = torch.zeros(batch_size, 4, 4)
        for b in range(batch_size):
            T[b, ::] = torch.Tensor(
                cls.spherical_marker_transform(
                    radius[b], azimuth[b], zenith[b], input_degrees
                )
            )
        return T

    def forward(
        self,
        source_image: torch.Tensor,
        transforms: torch.Tensor,
        canvas_size: Tuple[int, int],
    ):

        square_size = source_image.shape[2:4]
        warped_img = self.warp_image(
            source_image=source_image,
            transforms=transforms,
            canvas_size=canvas_size,
        )
        unwarped_img = self.unwarp_image(
            canvas_image=warped_img,
            transforms=transforms,
            square_size=square_size,
        )

        return unwarped_img
