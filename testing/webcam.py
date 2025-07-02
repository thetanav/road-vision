import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("../yolov5/yolov8n.pt")

# Real-time processing with overlay info
cap = cv2.VideoCapture("./videos/dashcam2.mp4")  # or video file


def detect_lane_lines(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection - tuned for lane markings
    edges = cv2.Canny(blur, 50, 150)

    # Create region of interest - trapezoid shape from bottom
    height, width = frame.shape[:2]

    # Define the region where lanes typically appear
    region_points = np.array(
        [
            [0, height],  # Bottom left
            [width // 2 - 60, height * 0.6],  # Top left (adjust 0.6 for how far up)
            [width // 2 + 60, height * 0.6],  # Top right
            [width, height],  # Bottom right
        ],
        np.int32,
    )

    # Create mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [region_points], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,  # Distance resolution
        theta=np.pi / 180,  # Angle resolution
        threshold=50,  # Min votes
        minLineLength=40,  # Min line length
        maxLineGap=100,  # Max gap between line segments
    )

    return lines, region_points


def draw_lane_lines(frame, lines):
    if lines is not None:
        # Separate left and right lanes
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope to separate left/right lanes
            if slope < -0.5:  # Left lane (negative slope)
                left_lines.append(line[0])
            elif slope > 0.5:  # Right lane (positive slope)
                right_lines.append(line[0])

        # Draw lanes
        lane_frame = frame.copy()

        # Draw left lane (blue)
        for line in left_lines:
            x1, y1, x2, y2 = line
            cv2.line(lane_frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

        # Draw right lane (red)
        for line in right_lines:
            x1, y1, x2, y2 = line
            cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        return lane_frame

    return frame


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

    threshold = 100  # Object too close
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

    cv2.imshow("Smart Dashcam - Objects + Lanes", final_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
