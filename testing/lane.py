import cv2
import numpy as np


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 180, 240)
    height, width = edges.shape[:2]
    mask = np.zeros_like(edges)
    if len(edges.shape) > 2:
        channel_count = edges.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    bottom_left = [width * 0.15, height]
    top_left = [width * 0.45, height * 0.6]
    bottom_right = [width * 0.95, height]
    top_right = [width * 0.55, height * 0.6]
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, ver, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # masker_img !
    # Focus on the lower half of the frame
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, 20, np.array([]), minLineLength=20, maxLineGap=180
    )
    line_img = np.zeros_like(frame)
    line_img = slope_lines(line_img, lines)
    # houghed_img !
    return cv2.addWeighted(frame, 0.8, line_img, 0.2, 0)


def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []
    right_lines = []

    # Handle case when no lines are detected
    if lines is None:
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                pass  # verticle line
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:  # check if > works
                    right_lines.append((m, c))

    # Only calculate means if we have lines
    if len(left_lines) > 0:
        left_lines = np.mean(left_lines, axis=0)
    else:
        left_lines = None

    if len(right_lines) > 0:
        right_lines = np.mean(right_lines, axis=0)
    else:
        right_lines = None

    # Only process lines that exist
    valid_lines = []
    if left_lines is not None:
        valid_lines.append(left_lines)
    if right_lines is not None:
        valid_lines.append(right_lines)

    for slope, intercept in valid_lines:
        # Skip if slope is zero or very small to avoid division by zero
        if abs(slope) < 0.001:
            continue

        # getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1 = int(rows)  # image.shape[0]

        # taking y2 upto 60% of actual height or 60% of y1
        y2 = int(rows * 0.6)  # int(0.6*y1)

        # we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

            # Check if coordinates are reasonable (within image bounds)
            if -1000 <= x1 <= cols + 1000 and -1000 <= x2 <= cols + 1000:
                poly_vertices.append((x1, y1))
                poly_vertices.append((x2, y2))
                draw_lines(img, np.array([[[x1, y1, x2, y2]]]))
        except (ZeroDivisionError, OverflowError, ValueError):
            # Skip this line if there are mathematical errors
            continue

    # Only fill polygon if we have enough vertices
    if len(poly_vertices) >= 4:
        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array([poly_vertices], "int32"), color=(0, 255, 0))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.0)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
