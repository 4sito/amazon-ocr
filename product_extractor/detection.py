# product_extractor/detection.py
from typing import List, Tuple
import cv2
import numpy as np

Box = Tuple[int, int, int, int]          # x,y,w,h
TextBox = Tuple[int, int, int, int, str] # x,y,w,h,text

def detect_colored_regions(img, sat_thresh: int = 40, area_thresh: int = 5000) -> List[Box]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= area_thresh:
            boxes.append((x, y, w, h))
    return boxes

def assign_text_to_images(text_boxes: List[TextBox], image_boxes: List[Box], row_gap: int = 50) -> List[List[TextBox]]:
    rows = []
    image_boxes = sorted(image_boxes, key=lambda b: b[1])
    text_boxes = sorted(text_boxes, key=lambda b: b[1])

    for i, (x, y, w, h) in enumerate(image_boxes):
        row_top = y + h
        row_bottom = image_boxes[i + 1][1] if i + 1 < len(image_boxes) else float("inf")
        row_text = [
            b for b in text_boxes
            if (row_top - row_gap) <= (b[1] + b[3] // 2) <= row_bottom
        ]
        if row_text:
            rows.append(row_text)
    return rows

def group_text_boxes_proximity(img_shape, text_boxes: List[TextBox], x_threshold: int = 100, y_threshold: int = 100):
    if not text_boxes:
        return []

    sorted_boxes = sorted(text_boxes, key=lambda b: (b[1], b[0]))
    groups = []
    current_group = []
    current_bbox = None
    height, width = img_shape[:2]

    for box in sorted_boxes:
        x, y, w, h, text = box
        center_x = x + w // 2
        center_y = y + h // 2

        belongs = False
        if current_group and current_bbox:
            min_gx, min_gy, max_gx, max_gy = current_bbox
            if (min_gx - x_threshold <= center_x <= max_gx + x_threshold and
                min_gy - y_threshold <= center_y <= max_gy + y_threshold):
                belongs = True

        if belongs:
            current_group.append(box)
            min_gx = min(b[0] for b in current_group)
            max_gx = max(b[0] + b[2] for b in current_group)
            min_gy = min(b[1] for b in current_group)
            max_gy = max(b[1] + b[3] for b in current_group)
            current_bbox = (min_gx, min_gy, max_gx, max_gy)
        else:
            if current_group:
                min_x = max(0, min(b[0] for b in current_group) - 10)
                max_x = min(width, max(b[0] + b[2] for b in current_group) + 10)
                min_y = max(0, min(b[1] for b in current_group) - 10)
                max_y = min(height, max(b[1] + b[3] for b in current_group) + 10)
                total_chars = sum(len(b[4]) for b in current_group)
                groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))
            current_group = [box]
            current_bbox = (x, y, x + w, y + h)

    if current_group:
        min_x = max(0, min(b[0] for b in current_group) - 10)
        max_x = min(width, max(b[0] + b[2] for b in current_group) + 10)
        min_y = max(0, min(b[1] for b in current_group) - 10)
        max_y = min(height, max(b[1] + b[3] for b in current_group) + 10)
        total_chars = sum(len(b[4]) for b in current_group)
        groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))

    return groups

def detect_product_columns_in_row(row_img, text_boxes_in_row: List[TextBox], min_column_width: int = 200, gap_threshold: int = 50):
    if not text_boxes_in_row:
        return [(0, 0, row_img.shape[1], row_img.shape[0])]

    x_positions = []
    for box in text_boxes_in_row:
        x, _, w, _, _ = box
        x_positions.extend([x, x + w])

    x_positions = sorted(set(x_positions))
    gaps = []
    for i in range(len(x_positions) - 1):
        start = x_positions[i]
        end = x_positions[i + 1]
        gap_w = end - start
        has_text = any(b[0] < end and b[0] + b[2] > start for b in text_boxes_in_row)
        if not has_text and gap_w >= gap_threshold:
            gaps.append((start, end, gap_w))

    columns = []
    row_h = row_img.shape[0]
    row_w = row_img.shape[1]
    if not gaps:
        columns.append((0, 0, row_w, row_h))
    else:
        gaps = sorted(gaps, key=lambda g: g[0])
        prev_end = 0
        for gap_start, gap_end, _ in gaps:
            if gap_start - prev_end >= min_column_width:
                columns.append((prev_end, 0, gap_start - prev_end, row_h))
            prev_end = gap_end
        if row_w - prev_end >= min_column_width:
            columns.append((prev_end, 0, row_w - prev_end, row_h))
    return columns
