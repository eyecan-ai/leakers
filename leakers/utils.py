from typing import Sequence, Union
from esurface.geometry.transforms import Transforms3DUtils
import cv2
import numpy as np


class TransformsUtils:
    @classmethod
    def transform_to_opencv(cls, T: np.ndarray):
        rot = T[:3, :3]
        tvec = T[:3, 3]
        rvec, _ = cv2.Rodrigues(rot)
        return rvec, tvec

    @classmethod
    def opencv_to_transform(cls, rvec, tvec):
        rot, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = tvec.ravel()
        return T

    @classmethod
    def square_points_3D(cls, square_size: float):
        return (
            np.array(
                [
                    [-1, 1, 0],
                    [1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                ]
            )
            * square_size
            * 0.5
        )


class AugmentedReality2DUtils(object):
    @classmethod
    def target_as_point(cls, camera_to_target, camera_matrix, dist_coeffs=None):
        dist_coeffs = (
            dist_coeffs
            if dist_coeffs is not None
            else np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        rvec, _ = cv2.Rodrigues(camera_to_target[:3, :3])
        tvec = camera_to_target[:3, 3]

        obj_points = np.array([0.0, 0.0, 0.0]).reshape((3, -1))
        image_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        image_points = image_points.ravel()
        image_points = image_points.astype(int)
        return image_points

    @classmethod
    def draw_target_as_point(
        cls,
        image,
        camera_to_target,
        camera_matrix,
        dist_coeffs=None,
        color=(0, 255, 255),
        radius=5,
        thickness=3,
    ):

        dist_coeffs = (
            dist_coeffs
            if dist_coeffs is not None
            else np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        image = image.copy()
        image_points = cls.target_as_point(camera_to_target, camera_matrix, dist_coeffs)
        if image_points[0] >= 0 and image_points[1] >= 0:
            if image_points[0] < image.shape[1] and image_points[1] < image.shape[0]:
                cv2.circle(
                    image,
                    (image_points[0], image_points[1]),
                    radius,
                    color,
                    thickness=thickness,
                )
        return image

    @classmethod
    def draw_target_as_rf(
        cls,
        image,
        camera_to_target,
        camera_matrix,
        dist_coeffs=None,
        color=(0, 255, 255),
        radius=5,
        thickness=3,
        axis_length=0.01,
        min_frustum_similarity=0,
    ):

        dist_coeffs = (
            dist_coeffs
            if dist_coeffs is not None
            else np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        image = image.copy()

        object_center = camera_to_target[:3, 3].copy()
        object_center /= np.linalg.norm(object_center)
        frustum_similarity = np.dot(object_center, np.array([0.0, 0.0, 1.0]))

        if frustum_similarity > min_frustum_similarity:
            tip_x = np.dot(
                camera_to_target, Transforms3DUtils.translation([axis_length, 0, 0])
            )
            tip_y = np.dot(
                camera_to_target, Transforms3DUtils.translation([0, axis_length, 0])
            )
            tip_z = np.dot(
                camera_to_target, Transforms3DUtils.translation([0, 0, axis_length])
            )
            tip_x = cls.target_as_point(tip_x, camera_matrix, dist_coeffs)
            tip_y = cls.target_as_point(tip_y, camera_matrix, dist_coeffs)
            tip_z = cls.target_as_point(tip_z, camera_matrix, dist_coeffs)

            image_points = cls.target_as_point(
                camera_to_target, camera_matrix, dist_coeffs
            )
            if cls.is_point_in_image(image, image_points):
                if (
                    cls.is_point_in_image(image, tip_x)
                    and cls.is_point_in_image(image, tip_y)
                    and cls.is_point_in_image(image, tip_z)
                ):
                    p = (image_points[0], image_points[1])
                    cv2.circle(image, p, radius, color, thickness=thickness)
                    cv2.line(
                        image,
                        p,
                        tuple(tip_x),
                        (255, 0, 0),
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.line(
                        image,
                        p,
                        tuple(tip_y),
                        (0, 255, 0),
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.line(
                        image,
                        p,
                        tuple(tip_z),
                        (0, 0, 255),
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
        return image

    @classmethod
    def is_point_in_image(cls, image: np.ndarray, point: np.ndarray):
        if point[0] >= 0 and point[1] >= 0:
            if point[0] < image.shape[1] and point[1] < image.shape[0]:
                return True
        return False

    @classmethod
    def bounding_box_corners(
        cls, camera_to_target: np.ndarray, size: np.ndarray
    ) -> Sequence:
        """Generates 3D bounding box corners given centered reference frame and size in meters

        :param camera_to_target: centered reference frame
        :type camera_to_target: np.ndarray
        :param size: box size in meters
        :type size: np.ndarray
        :return: list of ordered 3d corners
        :rtype: Sequence
        """

        sx, sy, sz = size
        box_mat = np.array(
            [
                [+1, +1, -1, -1, +1, +1, -1, -1],
                [-1, +1, +1, -1, -1, +1, +1, -1],
                [+1, +1, +1, +1, -1, -1, -1, -1],
            ]
        )
        box_mat = 0.5 * box_mat
        box_mat = np.transpose(box_mat)

        size_mat = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
        points = np.matmul(box_mat, size_mat)

        out_points = []

        for row in points:
            ph = np.array([row[0], row[1], row[2], 1.0]).reshape((4, 1))
            ph = np.dot(camera_to_target, ph)
            T = np.eye(4)
            T[:3, 3] = ph[:3].ravel()
            out_points.append(T)
        return out_points

    @classmethod
    def draw_3d_bounding_box(
        cls,
        image: np.ndarray,
        camera_to_target: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Union[None, np.ndarray],
        object_size: Sequence[float] = [0.1, 0.1, 0.1],
        color: Sequence = (255, 0, 255),
        thickness: int = 2,
        draw_axes: bool = True,
    ) -> np.ndarray:
        """Draw a 3D bounding box given camera to target reference frame

        :param image: output image
        :type image: np.ndarray
        :param camera_to_target: camera to target reference frame
        :type camera_to_target: np.ndarray
        :param camera_matrix: camera matrix
        :type camera_matrix: np.ndarray
        :param dist_coeffs: camera dist coefficients
        :type dist_coeffs: Union[None, np.ndarray]
        :param object_size: bounding box size in meters, defaults to [0.1, 0.1, 0.1]
        :type object_size: Sequence[float], optional
        :param color: wireframe color, defaults to (255, 0, 255)
        :type color: Sequence, optional
        :param thickness: wireframe thickness, defaults to 2
        :type thickness: int, optional
        :param draw_axes: TRUE to draw 3D axes, defaults to True
        :type draw_axes: bool, optional
        :return: resulting image
        :rtype: np.ndarray
        """

        image = image.copy()
        box_targets = cls.bounding_box_corners(camera_to_target, object_size)
        box_points = []
        for box_target in box_targets:
            # camera_to_kp = np.dot(np.linalg.inv(camera_pose), box_target)
            points = cls.target_as_point(
                box_target,
                camera_matrix,
                dist_coeffs
                if dist_coeffs is not None
                else np.array([0.0, 0.0, 0.0, 0.0]),
            )
            box_points.append(points)

        cls._draw_3d_bounding_box_wireframe(box_points, image)
        return image

    @classmethod
    def _draw_3d_bounding_box_wireframe(
        cls,
        points: Sequence,
        output: np.ndarray,
        color: Sequence = (255, 0, 255),
        thickness=2,
        draw_axes: bool = True,
    ):
        """Draw 3d box wirefreame and axes from ordered list of 2D bounding box corners

        :param points: ordered 2D bounding box corners
        :type points: Sequence
        :param output: output image
        :type output: np.ndarray
        :param color: wireframe color, defaults to (255, 0, 255)
        :type color: tuple, optional
        :param thickness: wireframe thickness, defaults to 2
        :type thickness: int, optional
        :param draw_axes: TRUE to draw 3D axes
        :type draw_axes: bool, optional
        """

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[1]),
            tuple(points[2]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[2]),
            tuple(points[3]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[3]),
            tuple(points[0]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            output,
            tuple(points[4]),
            tuple(points[5]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[5]),
            tuple(points[6]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[6]),
            tuple(points[7]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[7]),
            tuple(points[4]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[4]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[1]),
            tuple(points[5]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[2]),
            tuple(points[6]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple(points[3]),
            tuple(points[7]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        # X
        cv2.line(
            output,
            tuple(points[7]),
            tuple(points[4]),
            (255, 0, 0),
            thickness=thickness + 2,
            lineType=cv2.LINE_AA,
        )
        # Z
        cv2.line(
            output,
            tuple(points[7]),
            tuple(points[3]),
            (0, 0, 255),
            thickness=thickness + 2,
            lineType=cv2.LINE_AA,
        )
        # Y
        cv2.line(
            output,
            tuple(points[6]),
            tuple(points[7]),
            (0, 255, 0),
            thickness=thickness + 2,
            lineType=cv2.LINE_AA,
        )
