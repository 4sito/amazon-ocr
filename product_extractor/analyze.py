# product_extractor/analyze.py
from typing import Dict
import os
from PIL import Image, ImageDraw
import cv2
from .commons import upscale_image
from .ocr_utils import get_text_boxes, extract_price, extract_unit_price, extract_rating, extract_reviews_count, extract_delivery_info

def analyze_product_image(product_img, product_id: int, debug_dir: str, min_confidence: int = 30) -> Dict:
    """Analyze a product image and return a dictionary of extracted fields."""
    print(f"\n  Analyzing product {product_id}...")
    upscaled = upscale_image(product_img, scale_factor=2.0)
    text_boxes = get_text_boxes(upscaled, min_confidence=min_confidence)

    full_text = " ".join([b[4] for b in text_boxes])
    print(f"    Full OCR text: {full_text}")

    data = {
        'id_utente': '',
        'ricerca': '',
        'ordinamento': product_id,
        'sponsorizzato': '',
        'scelta_amazon': '',
        'prezzo': extract_price(full_text),
        'prezzo_per_unita': extract_unit_price(full_text),
        'unita_misura': (extract_unit_price(full_text).split('/')[-1] if '/' in extract_unit_price(full_text) else ''),
        'prezzo_consegna': extract_delivery_info(full_text),
        'valutazione_media': extract_rating(full_text),
        'n_recensioni': extract_reviews_count(full_text),
        'login': ''
    }

    # Build debug image with boxes (PIL)
    debug_img = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(debug_img)
    for (x, y, w, h, text) in text_boxes:
        label = ''
        if '€' in text:
            label = 'PREZZO'
        elif '(' in text and ')' in text and text.strip().startswith('('):
            label = 'REVIEWS'
        elif ',' in text and len(text) <= 4 and text[0].isdigit():
            label = 'RATING'
        elif 'consegna' in text.lower():
            label = 'DELIVERY'
        if label:
            draw.text((x, max(0, y - 15)), label, fill='red')
        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"product_{product_id}_debug.png")
    debug_img.save(debug_path)
    print(f"    Extracted data: {data}")
    print(f"    Debug image saved: {debug_path}")

    return data
