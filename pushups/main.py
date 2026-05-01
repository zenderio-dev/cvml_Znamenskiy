import cv2
import time
import threading
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound


BASE = Path(__file__).resolve().parent
MODEL = BASE / "yolo26n-pose.pt"
SOUND = BASE / "acolyteyes2.mp3"

LS, RS, LE, RE, LW, RW = 5, 6, 7, 8, 9, 10
DOWN_ANGLE = 105
UP_ANGLE = 155
RESET_TIME = 3
CONF = 0.25


def play():
    threading.Thread(target=lambda: playsound(str(SOUND)), daemon=True).start()


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def get_person(result):
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()

    if len(boxes) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return kpts[np.argmax(areas)]


def pushup_angle(k):
    angles = []

    if k[LS][2] > CONF and k[LE][2] > CONF and k[LW][2] > CONF:
        angles.append(angle(k[LS][:2], k[LE][:2], k[LW][:2]))

    if k[RS][2] > CONF and k[RE][2] > CONF and k[RW][2] > CONF:
        angles.append(angle(k[RS][:2], k[RE][:2], k[RW][:2]))

    return np.mean(angles) if angles else None


def put(img, text, y, color=(0, 255, 0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


model = YOLO(str(MODEL))
cap = cv2.VideoCapture(0)

count = 0
stage = "up"
last_seen = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, verbose=False)[0]
    k = get_person(result)

    annotator = Annotator(frame)

    if k is None:
        if time.time() - last_seen > RESET_TIME:
            count = 0
            stage = "up"

        img = annotator.result()
        put(img, "No person", 80, (0, 0, 255))

    else:
        last_seen = time.time()

        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        img = annotator.result()

        a = pushup_angle(k)

        if a is not None:
            if a < DOWN_ANGLE and stage == "up":
                stage = "down"

            if a > UP_ANGLE and stage == "down":
                stage = "up"
                count += 1
                play()

            put(img, f"Angle: {int(a)}", 80, (255, 255, 255))

    put(img, f"Push-ups: {count}", 40)
    put(img, f"Stage: {stage}", 120, (0, 255, 255))

    cv2.imshow("Push-ups", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()