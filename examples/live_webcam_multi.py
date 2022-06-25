import cv2
import numpy as np
from leakers.detectors.factory import LeakersDetectorsFactory

PAD = lambda x, padding, color: cv2.copyMakeBorder(
    x,
    top=padding,
    bottom=padding,
    left=padding,
    right=padding,
    borderType=cv2.BORDER_CONSTANT,
    value=[color, color, color],
)

checkpoint = "/tmp/leakers/leaker_code/version_2/checkpoints/last.ckpt"
detector = LeakersDetectorsFactory.create_from_checkpoint(filename=checkpoint)

leakers = detector.generate_raw_leakers()

mosaic_0 = np.hstack((leakers[0]["image"], leakers[1]["image"]))
mosaic_1 = np.hstack((leakers[2]["image"], leakers[3]["image"]))
mosaic = np.vstack((mosaic_0, mosaic_1))
mosaic = PAD(PAD(mosaic, 5, 0), 32, 255)

cv2.imshow(f"Mosaic", mosaic)
cv2.waitKey(1)

cam = cv2.VideoCapture("http://192.168.1.4:4747/video")
while True:
    ret_val, img = cam.read()
    if not ret_val:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect(img)

    for d in detections:
        img = detector.draw_detection(img, d)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("my webcam", img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
