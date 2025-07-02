import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("../yolov5/yolov8n.pt")

# Real-time processing with overlay info
cap = cv2.VideoCapture("./videos/dashcam2.mp4")  # or video file


# --- Lane detection pipeline ---
def lane_detection_pipeline(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # Define region of interest: middle of the frame, above a certain distance from the bottom (e.g., 80 pixels)
    bottom_offset = 80  # pixels from the bottom to avoid the car's bonnet
    polygon = np.array(
        [
            [
                (int(0.35 * width), height - bottom_offset),
                (int(0.35 * width), int(0.6 * height)),
                (int(0.65 * width), int(0.6 * height)),
                (int(0.65 * width), height - bottom_offset),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50
    )
    line_image = np.zeros_like(frame)
    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Keep only lines that are vertical or slanted (not horizontal)
            if 20 < abs(angle) < 160:  # Exclude near-horizontal lines
                filtered_lines.append((x1, y1, x2, y2))
        # Limit the number of lines drawn (e.g., max 8)
        for x1, y1, x2, y2 in filtered_lines[:8]:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 8)
    return line_image


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

    threshold = 0.0005  # Object too close
    # Alert for close objects
    for box in results[0].boxes:
        # If bounding box is large (object is close)
        box_area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
        if box_area > threshold:  # Object too close
            cv2.putText(
                frame,
                "WARNING: CLOSE OBJECT!",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

    # lane_lines = lane_finding_pipeline(frame)
    lane_lines = lane_detection_pipeline(frame)
    # Overlay lane lines on the annotated frame
    final_frame = cv2.addWeighted(annotated_frame, 0.8, lane_lines, 1, 0)

    cv2.imshow("Smart Dashcam - Objects + Lanes", final_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
