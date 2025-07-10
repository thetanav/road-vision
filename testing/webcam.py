import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from lane import detect_lanes
from playsound import playsound

model = YOLO("../model.pt")

cap = cv2.VideoCapture("./videos/dashcam3.mp4")

last_beep_time = 0


def beep():
    playsound("../sounds/warning.mp3")


def get_color(class_id):
    np.random.seed(class_id)
    color = np.random.randint(0, 255, 3)
    return tuple(int(c) for c in color)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lane detection overlay
    frame_with_lanes = detect_lanes(frame)

    results = model(frame, conf=0.5)

    annotated_frame = frame_with_lanes.copy()

    if results[0].boxes is not None:
        threshold = 800  # Object too close
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = abs((x2 - x1) * (y2 - y1))
            frame_height, frame_width = frame.shape[:2]
            center_left = int(frame_width * 1 / 3) + 40
            center_right = int(frame_width * 2 / 3) + 40
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            now = time.time()

            if (
                class_name == "car"
                and box_area > threshold * 10
                and abs(y2 - y1) / abs(x2 - x1) > 0.5
                and center_left <= box_center_x <= center_right
            ):
                cv2.putText(
                    annotated_frame,
                    "CAR TOO CLOSE!",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                if time.time() - last_beep_time > 1:
                    threading.Thread(target=beep, daemon=True).start()
                    last_beep_time = time.time()
            elif class_name == "person" and box_area > threshold * 3:
                cv2.putText(
                    annotated_frame,
                    "PERSON TOO CLOSE!",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                if time.time() - last_beep_time > 1:
                    threading.Thread(target=beep, daemon=True).start()
                    last_beep_time = time.time()

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), get_color(class_id), 2)
            label = f"{class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - th - 4),
                (x1 + tw, y1),
                get_color(class_id),
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    window_name = "Smart Dashcam System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated_frame)

    if cv2.getWindowProperty("Smart Dashcam System", cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
