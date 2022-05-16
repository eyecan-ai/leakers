from pathlib import Path
import click
import numpy as np
import cv2
import threading
import time


@click.command("live", help="Compile Configuration file")
@click.option("-m", "--model", required=True, help="Leakers Model File.")
@click.option(
    "-s",
    "--source",
    default="http://192.168.1.5:4747/video",
    help="Stream source [OpenCV VideoCapture url].",
)
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU Training")
def live(
    model: str,
    source: str,
    cuda: bool,
):

    import cv2
    import cv2
    from leakers.detectors.factory import LeakersDetectorsFactory
    import time

    device = "cuda" if cuda else "cpu"
    detector = LeakersDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )
    leakers = detector.generate_leakers(border=2)

    for leaker in leakers:
        leaker_id, leaker_image = leaker["id"], leaker["image"]
        cv2.namedWindow(f"L{leaker_id}", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(f"L{leaker_id}", cv2.cvtColor(leaker_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    class CamBuffer:
        def __init__(self, source: str) -> None:
            self._source = source
            self._active = True
            self._last_frame = None
            self._camera = cv2.VideoCapture(source)
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

        def stop(self):
            self._active = False

        def frame(self):
            return self._last_frame

        def loop(self):
            while self._active:
                ret_val, img = self._camera.read()
                self._last_frame = img
                if not ret_val:
                    self._active = False
                time.sleep(0.01)

    cam = CamBuffer(source)

    while True:

        img = cam.frame()
        if img is None:
            continue

        H, W = img.shape[:2]
        K = np.array(([500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1])).reshape((3, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_3d(img, marker_size=0.1, camera_matrix=K)

        for d in detections:
            img = detector.draw_detection(img, d)
            img = detector.draw_detection_3d(img, d, camera_matrix=K, marker_size=0.1)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("live", img)
        if cv2.waitKey(1) == 27:
            cam.stop()
            break  # esc to quit
