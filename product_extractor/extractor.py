# product_extractor/extractor.py
from typing import List, Dict
import os
from datetime import datetime
import pandas as pd
from .commons import init_dir
from .pdf_utils import pdf_to_image
from .ocr_utils import get_text_boxes
from .detection import detect_colored_regions, assign_text_to_images, group_text_boxes_proximity, detect_product_columns_in_row
from .analyze import analyze_product_image

def extract_products_from_pdf(
    pdf_path: str,
    output_dir: str = "output_products_data",
    x_threshold: int = 400,
    y_threshold: int = 100,
    min_chars: int = 400,
    min_confidence: int = 30
) -> None:
    base_dir = os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf', ''))
    init_dir(base_dir)
    debug_dir = os.path.join(base_dir, "debug_images")
    init_dir(debug_dir)

    try:
        import fitz
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"Failed to open PDF {pdf_path}: {e}")
        return

    print(f"Processing PDF {pdf_path} with {num_pages} page(s).")

    all_products: List[Dict] = []
    total_found = 0

    for page_num in range(num_pages):
        print(f"\n--- Processing Page {page_num + 1} ---")
        img = pdf_to_image(pdf_path, page_num)
        if img is None:
            continue

        text_boxes = get_text_boxes(img, min_confidence=min_confidence)
        image_boxes = detect_colored_regions(img, sat_thresh=40, area_thresh=30000)
        rows = assign_text_to_images(text_boxes, image_boxes, row_gap=50)

        grouped_boxes_with_chars = []
        for row in rows:
            row_groups = group_text_boxes_proximity(img.shape, row, x_threshold, y_threshold)
            grouped_boxes_with_chars.extend(row_groups)

        filtered_rows = [box for box in grouped_boxes_with_chars if box[4] >= min_chars]
        print(f"Found {len(filtered_rows)} product rows on page {page_num + 1}")

        for row_idx, (row_x, row_y, row_w, row_h, char_count) in enumerate(filtered_rows):
            row_img = img[row_y:row_y + row_h, row_x:row_x + row_w]

            # collect text boxes in this row (relative coords)
            text_boxes_in_row = []
            for tx, ty, tw, th, txt in text_boxes:
                if row_x <= tx <= row_x + row_w and row_y <= ty <= row_y + row_h:
                    text_boxes_in_row.append((tx - row_x, ty - row_y, tw, th, txt))

            product_columns = detect_product_columns_in_row(row_img, text_boxes_in_row, min_column_width=150, gap_threshold=50)

            for col_idx, (col_x, col_y, col_w, col_h) in enumerate(product_columns):
                total_found += 1
                product_img = row_img[col_y:col_y + col_h, col_x:col_x + col_w]
                product_path = os.path.join(base_dir, f"product_{total_found}.png")
                cv2_ok = False
                try:
                    import cv2
                    cv2_ok = cv2.imwrite(product_path, product_img)
                except Exception as e:
                    print(f"Failed to write product image: {e}")

                product_data = analyze_product_image(product_img, total_found, debug_dir, min_confidence=min_confidence)
                product_data.update({
                    'product_image_path': product_path,
                    'page': page_num + 1,
                    'row': row_idx,
                    'column': col_idx
                })
                all_products.append(product_data)

    if all_products:
        df = pd.DataFrame(all_products)
        column_order = [
            'id_utente', 'ricerca', 'ordinamento', 'sponsorizzato', 'scelta_amazon',
            'prezzo', 'prezzo_per_unita', 'unita_misura', 'prezzo_consegna',
            'valutazione_media', 'n_recensioni', 'login',
            'product_image_path', 'page', 'row', 'column'
        ]
        df = df.reindex(columns=[c for c in column_order if c in df.columns])
        excel_path = os.path.join(base_dir, f"products_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False)
        print("\n=== EXTRACTION COMPLETE ===")
        print(f"Total products extracted: {total_found}")
        print(f"Excel file created: {excel_path}")
        print(f"Product images saved in: {base_dir}")
        print(f"Debug images saved in: {debug_dir}")
    else:
        print("No products found!")
