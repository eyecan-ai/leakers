from typing import Dict, Sequence, Tuple
import numpy as np
import cv2
import torch
from leakers.datasets.alphabet import AlphabetDataset, BinaryAlphabetDataset
from leakers.datasets.datamodules import GenericAlphabetDatamodule

from leakers.nn.modules.base import LeakerModule
from leakers.utils import AugmentedReality2DUtils, TransformsUtils


class LeakersDetector(object):
    def __init__(
        self, model: LeakerModule, dataset: AlphabetDataset, grayscale: bool = False
    ):

        self.model = model
        self.dataset = dataset
        self.th_block_size = 11
        self.th_C = 2
        self.min_max_area_size = [200, 600 * 400]
        self.eps_curve = 0.01
        self.leaker_size = 64
        self._grayscale = grayscale

    def rectangle_info(self, points):
        ## TODO: TEMP DEBUG To detect Rectangle ratio?

        c0 = points[0, :]
        c1 = points[1, :]
        c2 = points[2, :]
        c3 = points[3, :]

        d0 = np.linalg.norm(c0 - c1)
        d1 = np.linalg.norm(c1 - c2)
        d2 = np.linalg.norm(c2 - c3)
        d3 = np.linalg.norm(c3 - c0)
        ratios = [d0 / d1, d0 / d2, d0 / d3]
        print("Info, RATIOS, ", ratios)

    def detect(self, image: np.ndarray) -> Sequence[Dict]:
        """Detect leakers from image

        :param image: input image
        :type image: np.ndarray
        :return: list of detected leakers
        :rtype: Sequence[Dict]
        """

        detections = self._build_detections(image, size=self.leaker_size)

        leakers_detections = []
        for detection in detections:
            rectangle, leaker, M = detection

            output = self.detect_single_leaker(leaker)
            if output is not None:
                code, rot = output["code"], output["rot"]
                points = rectangle.reshape((-1, 2))
                points = np.roll(points, rot, axis=0)
                self.rectangle_info(points)
                detection = {"code": code, "points": points, "image": leaker, "M": M}
                leakers_detections.append(detection)
        return leakers_detections

    def detect_3d(
        self, image: np.ndarray, marker_size: float, camera_matrix: np.ndarray
    ) -> Sequence[Dict]:

        leakers_detections = self.detect(image)
        points = TransformsUtils.square_points_3D(square_size=marker_size)

        for leaker_detection in leakers_detections:
            image_points = leaker_detection["points"]
            (_, rvec, tvec) = cv2.solvePnP(
                points.reshape(4, 1, 3),
                np.float32(image_points.reshape(4, 1, 2)),
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            leaker_detection["pose"] = TransformsUtils.opencv_to_transform(rvec, tvec)
        return leakers_detections

    def draw_detection(self, output_image: np.ndarray, detection: Dict) -> np.ndarray:

        output_image = output_image.copy()
        corner = detection["points"][0, :]
        cv2.drawContours(output_image, [detection["points"]], 0, (255, 255, 0), 3)
        cv2.circle(output_image, tuple(corner), 5, (0, 0, 255), -1)
        text = f"ID: {detection['code']}"
        cv2.putText(
            output_image,
            text,
            tuple(np.int32(corner - np.array([0, 20]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return output_image

    def draw_detection_3d(
        self,
        output_image: np.ndarray,
        detection: Dict,
        camera_matrix: np.ndarray,
        marker_size: float = 0.1,
    ):

        if "pose" in detection:
            output_image = output_image.copy()
            output_image = AugmentedReality2DUtils.draw_3d_bounding_box(
                output_image,
                detection["pose"],
                dist_coeffs=None,
                camera_matrix=camera_matrix,
                object_size=[marker_size, marker_size, marker_size],
            )
            output_image = AugmentedReality2DUtils.draw_target_as_rf(
                output_image,
                detection["pose"],
                dist_coeffs=None,
                camera_matrix=camera_matrix,
                axis_length=marker_size * 2,
            )
        return output_image

    def detect_single_leaker(self, leaker_image: np.ndarray):
        """
        Detects a single leaker in the image
        :param leake_image:
        :return:
        """

        if self._grayscale:
            leaker_image = cv2.cvtColor(leaker_image, cv2.COLOR_RGB2GRAY)

        if leaker_image.dtype == np.uint8:
            leaker_image = leaker_image / 255.0

        if self._grayscale:
            x = torch.Tensor(leaker_image).unsqueeze(0).unsqueeze(0)
        else:
            x = torch.Tensor(leaker_image).unsqueeze(0).permute(0, 3, 1, 2)

        x = x.repeat(4, 1, 1, 1)
        for rot in [0, 1, 2, 3]:
            x[rot, ::] = torch.rot90(x[rot, ::].unsqueeze(0), rot, dims=(2, 3)).squeeze(
                0
            )

        out = self.model.encode(x)
        code = out["code"]
        code_idx = self.dataset.words_to_indices(code.detach().cpu().numpy())
        rot = out["rot_classes"].detach().cpu().numpy()

        if np.unique(code_idx).size != 1:
            return None

        if 0 not in rot:
            return None

        rot_unroll = rot.copy()
        while rot_unroll[0] != 0:
            rot_unroll = np.roll(rot_unroll, 1)

        if not np.array_equal(rot_unroll, np.array([0, 1, 2, 3])):
            return None

        return {"code": code_idx[0], "rot": rot[0]}

    def detect_multi_leaker(self, leaker_image: np.ndarray, splits: int = 2):

        h, w = leaker_image.shape[:2]
        sh, sw = h // splits, w // splits

    def _detect_rectangles(self, input_image: np.ndarray) -> Sequence[np.ndarray]:
        """Detects rectangles in the image.

        :param input_image: input image
        :type input_image: np.ndarray
        :return: detected rectangles
        :rtype: Sequence[np.ndarray]
        """
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.th_block_size,
            self.th_C,
        )
        contours, hy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Searching through every region selected to
        # find the required polygon.
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Shortlisting the regions based on there area.
            if area > self.min_max_area_size[0] and area < self.min_max_area_size[1]:
                approx = cv2.approxPolyDP(
                    cnt, self.eps_curve * cv2.arcLength(cnt, True), True
                )
                if len(approx) == 4:
                    results.append(approx)
        return results

    def _build_detections(
        self, input_image: np.ndarray, size: int = 64
    ) -> Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute detections from squares within image. Each detection is
        a tuple (rectangle_points, warped_image, M2) where M2 is the inverse of the
        perspective transform that was used to warp the image.

        :param input_image: input image to detect
        :type input_image: np.ndarray
        :param size: resize to apply to warped images, defaults to 64
        :type size: int, optional
        :return: sequence of (rectangle_points, warped_image, M2)
        :rtype: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        """

        squares = self._detect_rectangles(input_image)
        detections = []
        size = size
        for index, s in enumerate(squares):
            points = np.array(s).reshape((4, 2)).astype(np.float32)

            dst = np.array(
                [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(points, dst)
            M2 = cv2.getPerspectiveTransform(dst, points)
            warp = cv2.warpPerspective(input_image, M, (size, size))
            padding = 3
            warp = warp[padding:-padding, padding:-padding]
            warp = cv2.resize(warp, (size, size), interpolation=cv2.INTER_CUBIC)
            detections.append((s, warp, M2.copy()))
        return detections

    def generate_raw_leakers(
        self, batch_size: int = 1
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Generate raw leakers as  ID,Image pairs.

        :param batch_size: batch size input for image generation, defaults to 1
        :type batch_size: int, optional
        :return: list of dict [{"id": id, "image": image}]
        :rtype: Sequence[Dict[str, np.ndarray]]
        """

        datamodule = GenericAlphabetDatamodule(
            dataset=self.dataset, batch_size=batch_size, drop_last=False
        )
        leakers = []
        for sample in datamodule.val_dataloader():

            imgs = self.model.generate(sample["x"])
            y = sample["y"]

            B, C, H, W = imgs.shape
            for b in range(B):
                leakers.append(
                    {
                        "id": y[b].item(),
                        "image": imgs[b, ::].permute(1, 2, 0).detach().cpu().numpy(),
                    }
                )

        return leakers

    def generate_leakers(
        self,
        output_size: int = 128,
        border: int = 1,
        padding: int = 16,
        batch_size: int = 1,
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Generate Leakers adding padding/border to raw leakers.

        :param output_size: desired output size, defaults to 128
        :type output_size: int, optional
        :param border: black border width, defaults to 1
        :type border: int, optional
        :param padding: white padding width, defaults to 16
        :type padding: int, optional
        :param batch_size: ibatch size input for image generation, defaults to 1
        :type batch_size: int, optional
        :return: list of dict [{"id": id, "image": image}]
        :rtype: Sequence[Dict[str, np.ndarray]]
        """

        PAD = lambda x, padding, color: cv2.copyMakeBorder(
            x,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[color, color, color],
        )

        raw_leakers = self.generate_raw_leakers(batch_size=batch_size)
        for raw_leaker in raw_leakers:
            raw_leaker["image"] = PAD(PAD(raw_leaker["image"], border, 0), padding, 255)
            raw_leaker["image"] = cv2.resize(
                raw_leaker["image"],
                (output_size, output_size),
                interpolation=cv2.INTER_LINEAR,
            )

        return raw_leakers
