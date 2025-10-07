# product_extractor/thresholds.py
"""
Compute good x/y grouping thresholds for your extractor.

Strategy (simple & deterministic):
 1. Try detecting colored/image regions (detect_colored_regions).
    - If found: median image-box height -> y_threshold (conservative fraction).
    - Compute column gaps per row -> x_threshold from median gap.
 2. Else: use OCR text boxes (get_text_boxes), cluster vertical centers into rows,
    compute median row height and column gaps similarly.
 3. Else: fallback to conservative fractions of page dimensions.

Returns a dict:
{
  "x_threshold": int(px),
  "y_threshold": int(px),
  "x_ratio": float(px / width),
  "y_ratio": float(px / height),
  "median_row_height": int or None,
  "median_col_gap": int or None,
  "source": "images"|"text"|"fallback"
}
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# relative imports to reuse your package modules
from .detection import detect_colored_regions
from .ocr_utils import get_text_boxes
from .pdf_utils import pdf_to_image

# small helpers
def _median(values: List[int]) -> Optional[int]:
    return None if not values else int(np.median(values))

def _cluster_1d(centers: List[int], gap_factor: float = 2.0) -> List[Tuple[int,int]]:
    """Simple 1D clustering: returns list of (min_center, max_center) ranges."""
    if not centers:
        return []
    centers = sorted(centers)
    diffs = [b - a for a, b in zip(centers[:-1], centers[1:])]
    if not diffs:
        return [(centers[0], centers[0])]
    med = np.median(diffs)
    split_thr = max(1.0, med * gap_factor)
    clusters = []
    current = [centers[0]]
    for c, d in zip(centers[1:], diffs):
        if d > split_thr:
            clusters.append((min(current), max(current)))
            current = [c]
        else:
            current.append(c)
    if current:
        clusters.append((min(current), max(current)))
    return clusters

def _median_gap_in_rows(rows: List[List[int]]) -> Optional[int]:
    """Given x-centers per row (list of lists), compute median neighbor gap across rows."""
    all_gaps = []
    for centers in rows:
        if len(centers) < 2:
            continue
        centers = sorted(centers)
        gaps = [b - a for a, b in zip(centers[:-1], centers[1:])]
        all_gaps.extend(gaps)
    return _median(all_gaps)

def compute_thresholds_from_image(
    img,
    prefer_images: bool = True,
    sat_thresh: int = 40,
    area_thresh: int = 2000,
    text_min_confidence: int = 30,
    conservative_frac: float = 0.6
):
    """
    Compute x/y thresholds from a rendered page image.

    Arguments:
      img: OpenCV BGR image (numpy array)
      prefer_images: if True prefer colored/image-detection results when available
      sat_thresh, area_thresh: passed to detect_colored_regions
      text_min_confidence: pytesseract min confidence for text boxes
      conservative_frac: fraction to reduce median sizes to avoid over-splitting

    Returns: dict described above
    """
    if img is None:
        raise ValueError("img is None")

    height, width = img.shape[:2]
    result = {
        "x_threshold": int(width * 0.5),
        "y_threshold": int(height * 0.12),
        "x_ratio": None,
        "y_ratio": None,
        "median_row_height": None,
        "median_col_gap": None,
        "source": "fallback"
    }

    # 1) image-based detection
    image_boxes = detect_colored_regions(img, sat_thresh=sat_thresh, area_thresh=area_thresh)
    if image_boxes and prefer_images:
        # median height and median width
        heights = [h for (_, _, _, h) in image_boxes]
        widths = [w for (_, _, w, _) in image_boxes]
        median_h = _median(heights)
        median_w = _median(widths)

        # group boxes into rows by y proximity
        # sort by top
        image_boxes_sorted = sorted(image_boxes, key=lambda b: (b[1], b[0]))
        rows_x_centers = []
        row_tol = max(4, int((median_h or height) / 3))
        current_row = []
        last_center_y = None
        for (x, y, w_box, h_box) in image_boxes_sorted:
            cx = x + w_box // 2
            cy = y + h_box // 2
            if last_center_y is None or abs(cy - last_center_y) <= row_tol:
                current_row.append(cx)
                last_center_y = cy if last_center_y is None else (last_center_y + cy) // 2
            else:
                rows_x_centers.append(sorted(current_row))
                current_row = [cx]
                last_center_y = cy
        if current_row:
            rows_x_centers.append(sorted(current_row))

        median_gap = _median_gap_in_rows(rows_x_centers)
        # compute thresholds conservatively
        y_threshold = max(10, int((median_h or height) * conservative_frac))
        if median_gap:
            x_threshold = max(30, int(median_gap * conservative_frac))
        else:
            # use approx column width based on median image width
            est_cols = max(1, int(round(width / (max(120, (median_w or width)))))
            )
            if est_cols <= 1:
                x_threshold = max(40, int(width * 0.5))
            else:
                approx_col_w = int(width / est_cols)
                x_threshold = max(30, int(approx_col_w * 0.5))

        result.update({
            "x_threshold": int(x_threshold),
            "y_threshold": int(y_threshold),
            "x_ratio": round(x_threshold / width, 3),
            "y_ratio": round(y_threshold / height, 3),
            "median_row_height": int(median_h) if median_h else None,
            "median_col_gap": int(median_gap) if median_gap else None,
            "source": "images"
        })
        print(result)
        return x_threshold, y_threshold

    # 2) OCR-based fallback
    text_boxes = get_text_boxes(img, min_confidence=text_min_confidence)
    if text_boxes:
        # compute vertical centers
        y_centers = [y + h // 2 for (x, y, w_box, h, _) in text_boxes]
        clusters = _cluster_1d(y_centers, gap_factor=2.0)
        row_heights = []
        rows_x_centers = []
        for (cmin, cmax) in clusters:
            # pick boxes whose center is inside [cmin-1, cmax+1]
            boxes_in_cluster = [(x, y, w_box, h, t) for (x, y, w_box, h, t) in text_boxes
                                if (y + h // 2) >= (cmin - 1) and (y + h // 2) <= (cmax + 1)]
            if not boxes_in_cluster:
                continue
            min_y = min(b[1] for b in boxes_in_cluster)
            max_y = max(b[1] + b[3] for b in boxes_in_cluster)
            row_heights.append(max_y - min_y)
            rows_x_centers.append(sorted([b[0] + b[2] // 2 for b in boxes_in_cluster]))

        median_row_h = _median(row_heights)
        median_gap = _median_gap_in_rows(rows_x_centers)

        y_threshold = max(10, int((median_row_h or height * 0.12) * conservative_frac))
        if median_gap:
            x_threshold = max(30, int(median_gap * conservative_frac))
        else:
            # estimate columns using row height
            est_cols = max(1, int(round(width / max(120, (median_row_h or height//4)))))
            if est_cols <= 1:
                x_threshold = int(width * 0.5)
            else:
                approx_col_w = int(width / est_cols)
                x_threshold = max(30, int(approx_col_w * 0.5))

        result.update({
            "x_threshold": int(x_threshold),
            "y_threshold": int(y_threshold),
            "x_ratio": round(x_threshold / width, 3),
            "y_ratio": round(y_threshold / height, 3),
            "median_row_height": int(median_row_h) if median_row_h else None,
            "median_col_gap": int(median_gap) if median_gap else None,
            "source": "text"
        })
        print(result)
        return x_threshold, y_threshold

    # 3) fallback heuristics
    y_threshold = max(10, int(height * 0.12))
    x_threshold = max(30, int(width * 0.5))
    result.update({
        "x_threshold": int(x_threshold),
        "y_threshold": int(y_threshold),
        "x_ratio": round(x_threshold / width, 3),
        "y_ratio": round(y_threshold / height, 3),
        "source": "fallback"
    })
    print(result)
    return x_threshold, y_threshold

def compute_thresholds_from_pdf(pdf_path: str, page: int = 0, dpi: int = 200, **kwargs) -> Dict:
    """
    Render a page and return compute_thresholds_from_image(img, **kwargs)
    """
    img = pdf_to_image(pdf_path, page, dpi=dpi)
    if img is None:
        raise RuntimeError(f"Could not render page {page} of {pdf_path}")
    return compute_thresholds_from_image(img, **kwargs)
