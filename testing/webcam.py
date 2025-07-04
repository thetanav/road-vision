import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("../model.pt")

# Real-time processing with overlay info
cap = cv2.VideoCapture("./videos/dashcam.mp4")  # or video file


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

    return lines, roi_points


def draw_lane_lines(frame, lanes, color=(0, 255, 0), thickness=5):
    line_image = np.zeros_like(frame)

    if lanes is not None:
        for line in lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    # Overlay lane lines on original frame
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return final_frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Count detections
    if results[0].boxes is not None:
        cars = sum(
            1 for box in results[0].boxes if model.names[int(box.cls[0])] == "car"
        )
        signs = sum(
            1 for box in results[0].boxes if "sign" in model.names[int(box.cls[0])]
        )

        cv2.putText(
            annotated_frame,
            f"Cars: {cars} | Signs: {signs}",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    threshold = 200  # Object too close
    # Alert for close objects
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        # Only warn if the object is a person and is too close
        if class_name == "person":
            box_area = (box.xyxy[0][2] - box.xyxy[0][0]) * (
                box.xyxy[0][3] - box.xyxy[0][1]
            )
            if box_area > threshold:
                cv2.putText(
                    annotated_frame,
                    "WARNING: CLOSE PERSON!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
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
