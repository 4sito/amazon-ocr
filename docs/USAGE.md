# Usage

## CLI
```
python -m product_extractor.cli my_catalog.pdf \
  --output out_dir \
  --min-chars 400 \
  --x-threshold 400 \
  --y-threshold 100 \
  --min-confidence 30
```

Outputs:
- `out_dir/my_catalog/` — product images
- `out_dir/my_catalog/debug_images/` — debug images
- `out_dir/my_catalog/products_data_YYYYMMDD_HHMMSS.xlsx` — Excel with extracted fields

## Programmatic
```py
from product_extractor.extractor import extract_products_from_pdf

extract_products_from_pdf(
    "catalog.pdf",
    output_dir="out",
    x_threshold=400,
    y_threshold=100,
    min_chars=400,
    min_confidence=30
)
```

## API surface (quick)
- `extract_products_from_pdf(pdf_path, output_dir="output_products_data", x_threshold=400, y_threshold=100, min_chars=400, min_confidence=30)`  
  Runs full pipeline. Side effects: saves files. Modify to return DataFrame if you prefer.

- `analyze_product_image(product_img, product_id, debug_dir, min_confidence=30)`  
  Returns dictionary of extracted fields for a single image (used internally).

- `get_text_boxes(img, min_confidence=30, lang='ita')` (pytesseract wrapper) — returns OCR bounding boxes.

## Tips
- Tune `min_chars`, `x_threshold`, `y_threshold`, `sat_thresh`, `area_thresh` per-catalog.
- Consider running `opencv-python-headless` on servers to avoid GUI dependencies.
