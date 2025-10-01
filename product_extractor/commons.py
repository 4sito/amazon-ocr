# product_extractor/common.py
from typing import Tuple
import os
import shutil
import cv2
import numpy as np

def clear_directory(folder: str) -> None:
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            # Keep side-effect printing minimal
            print(f'Failed to delete {file_path}. Reason: {e}')

def init_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    clear_directory(dir_path)

def upscale_image(img: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    """Upscale image for better OCR accuracy (pure transformation)."""
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Return a thresholded grayscale image to improve OCR."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
