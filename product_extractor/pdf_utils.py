# product_extractor/pdf_utils.py
from typing import Optional
import fitz  # PyMuPDF
import cv2
import numpy as np

def pdf_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Optional[np.ndarray]:
    """Convert specific page to OpenCV image. Returns None on failure."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        doc.close()
        return img
    except Exception as e:
        print(f"pdf_to_image error for {pdf_path} page {page_num}: {e}")
        return None
