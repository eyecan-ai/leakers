import cv2
from leakers.detectors.factory import LeakersDetectorsFactory
import time

checkpoint = "/tmp/leakers/leaker_alpha/version_3/checkpoints/epoch=1498-step=1498.ckpt"
detector = LeakersDetectorsFactory.create_from_checkpoint(filename=checkpoint)
leakers = detector.generate_leakers()

for leaker in leakers:
    leaker_id, leaker_image = leaker["id"], leaker["image"]
    cv2.namedWindow(f"L{leaker_id}", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(f"L{leaker_id}", leaker_image)
    cv2.waitKey(1)

cam = cv2.VideoCapture("http://192.168.1.3:4747/video")
while True:
    ret_val, img = cam.read()
    if not ret_val:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t1 = time.perf_counter()
    detections = detector.detect(img)
    t2 = time.perf_counter()
    # print(f"Time: {t2 - t1}\t Hz:{1./(t2-t1)}")

    for d in detections:
        img = detector.draw_detection(img, d)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("my webcam", img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
