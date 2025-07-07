import cv2
import numpy as np
from ultralytics import YOLO
import threading
from playsound import playsound
import time

model = YOLO("../model.pt")

# Real-time processing with overlay info
# Just put the video file here
# OR the dsashcam ip
cap = cv2.VideoCapture("./videos/dashcam2.mp4")
# cap = cv2.VideoCapture(0)

last_beep_time = 0  # Track last beep time


def beep():
    playsound("../sounds/tesla_warning_chime.mp3")


def detect_lane_lines(frame):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define Region of Interest
    height, width = edges.shape
    roi_points = np.array(
        [
            [
                (int(0.1 * width), height),
                (int(0.45 * width), int(0.6 * height)),
                (int(0.55 * width), int(0.6 * height)),
                (int(0.9 * width), height),
            ]
        ],
        dtype=np.int32,
    )

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_points, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100,
    )

    # Minimal: pick the most left or right line (edge of road)
    edge_line = None
    if lines is not None:
        # Find the line with the smallest x1 (leftmost) or largest x2 (rightmost)
        # Here, we pick the rightmost edge (largest x1/x2)
        rightmost_x = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            max_x = max(x1, x2)
            if max_x > rightmost_x:
                rightmost_x = max_x
                edge_line = line
    # Return as a list for compatibility
    return [edge_line] if edge_line is not None else None, roi_points


def draw_lane_lines(frame, lanes, color=(0, 255, 255), thickness=3):
    line_image = np.zeros_like(frame)
    if lanes is not None:
        for line in lanes:
            if line is not None:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return final_frame


def get_color(class_id):
    np.random.seed(class_id)
    color = np.random.randint(0, 255, 3)
    return tuple(int(c) for c in color)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Get annotated frame (custom minimal drawing)
    annotated_frame = frame.copy()

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

            # if not (center_left <= box_center_x <= center_right):
            #     continue

            now = time.time()

            # Close car detection with perspective
            if (
                class_name == "car"
                and box_area > threshold * 10
                and abs(y2 - y1) / abs(x2 - x1) > 0.5
                and center_left <= box_center_x <= center_right
            ):
                cv2.putText(
                    annotated_frame,
                    "CAR TOO CLOSE!",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
            elif class_name == "person" and box_area > threshold * 4:
                cv2.putText(
                    annotated_frame,
                    "PERSON TOO CLOSE!",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                if time.time() - last_beep_time > 3:
                    threading.Thread(target=beep, daemon=True).start()
                    last_beep_time = time.time()

            # Draw a thin, light green rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), get_color(class_id), 2)
            # Draw a small label
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

    # 2. Lane detection
    lanes, roi_points = detect_lane_lines(frame)
    final_frame = draw_lane_lines(annotated_frame, lanes)

    # Resize the frame to make the window small
    # Show the frame as-is, allow window to be resizable
    window_name = "Smart Dashcam - Objects + Lanes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, final_frame)

    # cv2.imshow("Smart Dashcam - Objects + Lanes", resized_frame)

    # Check if window was closed
    if (
        cv2.getWindowProperty("Smart Dashcam - Objects + Lanes", cv2.WND_PROP_VISIBLE)
        < 1
    ):
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
