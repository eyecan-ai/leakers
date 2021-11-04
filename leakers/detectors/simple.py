import numpy as np
import cv2
import torch
from leakers.datasets.alphabet import BinaryAlphabetDataset

from leakers.nn.modules.base import LeakerModule


class SimpleLeakerDetector(object):
    def __init__(
        self, model: LeakerModule, dataset: BinaryAlphabetDataset, device: str = "cuda"
    ):

        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.th_block_size = 11
        self.th_C = 2
        self.min_max_area_size = [200, 600 * 400]
        self.eps_curve = 0.01

    def detect_single_leaker(self, leaker_image: np.ndarray):
        """
        Detects a single leaker in the image
        :param leake_image:
        :return:
        """

        x = (
            torch.Tensor(leaker_image / 255.0)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        x = x.repeat(4, 1, 1, 1)
        for rot in [0, 1, 2, 3]:
            x[rot, ::] = torch.rot90(x[rot, ::].unsqueeze(0), rot, dims=(2, 3)).squeeze(
                0
            )

        out = self.model.encode(x)
        code = out["code"]
        code_idx = self.dataset.words_to_indices(code.detach().cpu().numpy())
        rot = out["rot_classes"].detach().cpu().numpy()

        # if np.unique(code_idx).size == 1:
        #     return None

        if 0 not in rot:
            return None

        rot_unroll = rot.copy()
        while rot_unroll[0] != 0:
            rot_unroll = np.roll(rot_unroll, 1)
            print(rot_unroll)

        # if not np.array_equal(rot_unroll, np.array([0, 1, 2, 3])):
        #     return None

        return {"code": code_idx[0], "rot": rot[0]}

    def detect_rectangles(self, input_image):
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

    def build_detections(self, input_image, size=256, approx_factor=0.1):
        squares = self.detect_rectangles(input_image)
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
