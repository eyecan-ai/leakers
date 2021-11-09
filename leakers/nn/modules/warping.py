import numpy as np
import torch
import kornia.geometry as kgeometry
from typing import Sequence, Tuple, Union
import transforms3d


class WarpingModule:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        transform: np.ndarray = np.eye(4),
        size: float = 0.05,
    ):

        self._camera_matrix = torch.Tensor(camera_matrix)
        self._size = size
        self._transform = torch.Tensor(transform).unsqueeze(0)
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
        self._points = torch.Tensor(self._points).unsqueeze(0)
        self._points = kgeometry.transform_points(self._transform, self._points)

    def points_2d(self):
        return kgeometry.camera.project_points(self._points, self._camera_matrix)

    def get_perspective_transform(self, source_image_size: Tuple[int, int]):
        W, H = source_image_size
        source_points = np.array(
            [
                [0, 0],
                [W, 0],
                [W, H],
                [0, H],
            ]
        )
        source_points = torch.Tensor(source_points).unsqueeze(0)
        target_points = self.points_2d()
        M = kgeometry.get_perspective_transform(source_points, target_points)
        M_inv = kgeometry.get_perspective_transform(target_points, source_points)
        return M, M_inv

    def warp_image(self, source_image: torch.Tensor, canvas_size: Tuple[int, int]):
        mH, mW = source_image.shape[2:4]
        H, W = canvas_size
        M, M_inv = self.get_perspective_transform([mH, mW])
        warped_img = kgeometry.warp_perspective(
            source_image, M, (H, W), align_corners=True
        )
        return warped_img

    def unwarp_image(self, canvas_image: torch.Tensor, square_size: Tuple[int, int]):
        H, W = canvas_image.shape[2:4]
        mH, mW = square_size
        M, M_inv = self.get_perspective_transform([mH, mW])
        unwarped_img = kgeometry.warp_perspective(
            canvas_image, M_inv, (mH, mW), align_corners=True
        )
        return unwarped_img

    @classmethod
    def spherical_marker_transform(
        cls, radius: float, azimuth: float, zenith: float, input_degrees: bool = True
    ):
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


class WarpingModuleV2:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        virtual_size: float = 0.05,
    ):

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
        B, _, _ = transforms.shape
        points = torch.Tensor(self._points).unsqueeze(0).repeat(B, 1, 1)
        t_points = kgeometry.transform_points(transforms, points)
        B, N, _ = t_points.shape

        t_points = t_points.view(B * N, 3)
        camera = self._camera_matrix.repeat(t_points.shape[0], 1, 1)
        p_points = kgeometry.camera.project_points(t_points, camera).view(B, N, 2)
        return p_points

    def get_perspective_transform(
        self, transforms: torch.Tensor, source_image_size: Tuple[int, int]
    ):
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
        source_points = torch.Tensor(source_points).unsqueeze(0).repeat(B, 1, 1)
        target_points = self.points_2d(transforms=transforms)
        M = kgeometry.get_perspective_transform(source_points, target_points)
        M_inv = kgeometry.get_perspective_transform(target_points, source_points)
        return M, M_inv

    def warp_image(
        self,
        source_image: torch.Tensor,
        transforms: torch.Tensor,
        canvas_size: Tuple[int, int],
    ):
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
    ):
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
    ):
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
    ):

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
