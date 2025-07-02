import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("../yolov5/yolov8n.pt")

# Real-time processing with overlay info
cap = cv2.VideoCapture("./videos/dashcam2.mp4")  # or video file


def lane_detection_pipeline(frame):
    # here use the lane_finding_pipeline function for lane detection
    pass


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
