# product_extractor/ocr_utils.py
from typing import List, Tuple, Dict
import pytesseract
import cv2
import re

TextBox = Tuple[int, int, int, int, str]  # x, y, w, h, text

def _parse_conf(conf_val) -> int:
    """Normalize tesseract conf value to int (-1 when unknown)."""
    try:
        return int(float(conf_val))
    except Exception:
        return -1

def get_text_boxes(img, min_confidence: int = 30, lang: str = 'ita') -> List[TextBox]:
    """Return list of (x,y,w,h,text) for boxes above min_confidence."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang=lang)

    boxes = []
    n = len(data.get('level', []))
    for i in range(n):
        conf = _parse_conf(data['conf'][i])
        if conf >= min_confidence:
            text = (data.get('text', [''])[i] or '').strip()
            if text:
                boxes.append((int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i]), text))
    return boxes

# Extraction helpers (pure functions)
def extract_price(text_content: str) -> str:
    """Return normalized price like '12,34€' or empty string."""
    # try patterns like 1.234,56 € or 1234,56€
    patterns = [
        r'(\d{1,3}(?:\.\d{3})*),(\d{2})\s*€',  # 1.234,56 €
        r'(\d{1,3}),(\d{2})\s*€'               # 123,45 €
    ]
    for p in patterns:
        m = re.search(p, text_content)
        if m:
            euros = m.group(1).replace('.', '')
            cents = m.group(2)
            return f"{euros},{cents}€"
    return ""

def extract_unit_price(text_content: str) -> str:
    m = re.search(r'\(([0-9.,]+€/[a-zA-Z%]+)\)', text_content)
    return m.group(1) if m else ""

def extract_rating(text_content: str) -> str:
    m = re.search(r'\b([0-5],[0-9])\b', text_content)
    return m.group(1) if m else ""

def extract_reviews_count(text_content: str) -> str:
    matches = re.findall(r'\((\d+)\)', text_content)
    return max(matches, key=lambda x: int(x)) if matches else ""

def extract_delivery_info(text_content: str) -> str:
    parts = text_content.split("Consegna")
    if len(parts) > 1:
        out = "Consegna" + parts[1].strip()
        return out[:200]
    return ""
