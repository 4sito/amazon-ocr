import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os, shutil
import pandas as pd
import re
from datetime import datetime

def calculate_dynamic_thresholds(img_shape, x_ratio=0.50, y_ratio=0.50):
    """
    Calculate thresholds as a percentage of image dimensions.
    
    Args:
        img_shape: (height, width, channels) from the image
        x_ratio: percentage of width (default 15%)
        y_ratio: percentage of height (default 5%)
    
    Returns:
        x_threshold, y_threshold
    """
    print(x_ratio)
    print(y_ratio)
    height, width = img_shape[:2]
    print(height)
    print(width)
    x_threshold = int(width * x_ratio)
    y_threshold = int(height * y_ratio)
    return x_threshold, y_threshold

def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def init_dir(dir):
    os.makedirs(dir, exist_ok=True)
    clear_directory(dir)

def upscale_image(img, scale_factor=2):
    """Upscale image for better OCR accuracy."""
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled

def preprocess_for_ocr(img):
    """Preprocess image for better OCR results."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def pdf_to_image(pdf_path, page_num, dpi=300):
    """Converts a specific page of a PDF to an OpenCV image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    doc.close()
    return img

def get_text_boxes(img, min_confidence=30):
    """Gets bounding boxes for all text elements using OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='ita')
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        if conf > min_confidence:
            text = data['text'][i].strip()
            if text:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                boxes.append((x, y, w, h, text))
    return boxes

def assign_text_to_images(text_boxes, image_boxes, row_gap=50):
    """Assigns OCR text boxes to rows defined by product images."""
    rows = []
    image_boxes = sorted(image_boxes, key=lambda b: b[1])
    text_boxes = sorted(text_boxes, key=lambda b: b[1])
    for i, (x, y, w, h) in enumerate(image_boxes):
        row_top = y + h
        if i + 1 < len(image_boxes):
            row_bottom = image_boxes[i+1][1]
        else:
            row_bottom = float("inf")
        row_text = [b for b in text_boxes if row_top - row_gap <= (b[1] + b[3]//2) <= row_bottom]
        if row_text:
            rows.append(row_text)
    return rows

def detect_product_columns_in_row(row_img, text_boxes_in_row, num_products=2):
    """
    Detects individual product columns within a product row based on expected number of products.
    
    Args:
        row_img: Image of the product row
        text_boxes_in_row: List of text boxes in the row
        num_products: Expected number of products per row (default 2)
    
    Returns:
        List of column bounding boxes (x, y, w, h)
    """
    row_height = row_img.shape[0]
    row_width = row_img.shape[1]
    
    # If no text boxes or single product, return full width
    if not text_boxes_in_row or num_products <= 1:
        return [(0, 0, row_width, row_height)]
    
    # Calculate expected column width and gap threshold based on num_products
    expected_column_width = row_width / num_products
    min_column_width = int(expected_column_width * 0.5)  # At least 50% of expected width
    gap_threshold = int(expected_column_width * 0.15)  # Gap should be ~15% of column width
    
    print(f"    Row width: {row_width}, Expected products: {num_products}")
    print(f"    Expected column width: {expected_column_width:.0f}, Min: {min_column_width}, Gap threshold: {gap_threshold}")
    
    # Get all x positions from text boxes
    x_positions = []
    for box in text_boxes_in_row:
        x, y, w, h, text = box
        x_positions.extend([x, x + w])
    x_positions = sorted(set(x_positions))
    
    # Find gaps between text regions
    gaps = []
    for i in range(len(x_positions) - 1):
        gap_start = x_positions[i]
        gap_end = x_positions[i + 1]
        gap_width = gap_end - gap_start
        
        # Check if there's text in this gap
        has_text_in_gap = any(box[0] < gap_end and box[0] + box[2] > gap_start 
                             for box in text_boxes_in_row)
        
        if not has_text_in_gap and gap_width >= gap_threshold:
            gaps.append((gap_start, gap_end, gap_width))
    
    # Build columns from gaps
    columns = []
    
    if not gaps:
        # No significant gaps found, divide equally
        print(f"    No gaps found, dividing equally into {num_products} columns")
        for i in range(num_products):
            col_x = int(i * expected_column_width)
            col_w = int(expected_column_width)
            # Adjust last column to reach end
            if i == num_products - 1:
                col_w = row_width - col_x
            columns.append((col_x, 0, col_w, row_height))
    else:
        # Use detected gaps to separate columns
        gaps = sorted(gaps, key=lambda g: g[0])
        print(f"    Found {len(gaps)} gaps")
        
        # Select the most significant gaps (up to num_products - 1)
        if len(gaps) > num_products - 1:
            # Sort by gap width and take the largest ones
            gaps = sorted(gaps, key=lambda g: g[2], reverse=True)[:num_products - 1]
            # Re-sort by position
            gaps = sorted(gaps, key=lambda g: g[0])
        
        prev_end = 0
        for gap_start, gap_end, gap_width in gaps:
            if gap_start - prev_end >= min_column_width:
                columns.append((prev_end, 0, gap_start - prev_end, row_height))
            prev_end = gap_end
        
        # Add final column
        if row_width - prev_end >= min_column_width:
            columns.append((prev_end, 0, row_width - prev_end, row_height))
    
    print(f"    Created {len(columns)} columns")
    return columns

def detect_colored_regions(img, sat_thresh=40, area_thresh=5000):
    """Detects colored regions (likely product images) in the page."""
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

def group_text_boxes_proximity(img_shape, text_boxes, x_threshold=100, y_threshold=100):
    """Groups text boxes based on their spatial proximity."""
    if not text_boxes:
        return []
    sorted_boxes = sorted(text_boxes, key=lambda b: (b[1], b[0]))
    groups = []
    current_group = []
    current_group_bbox = None
    height, width = img_shape[:2]
    for box in sorted_boxes:
        x, y, w, h, text = box
        box_center_x = x + w // 2
        box_center_y = y + h // 2
        belongs_to_group = False
        if current_group and current_group_bbox:
            min_gx, min_gy, max_gx, max_gy = current_group_bbox
            expanded_min_x = min_gx - x_threshold
            expanded_max_x = max_gx + x_threshold
            expanded_min_y = min_gy - y_threshold
            expanded_max_y = max_gy + y_threshold
            if (expanded_min_x <= box_center_x <= expanded_max_x and
                expanded_min_y <= box_center_y <= expanded_max_y):
                 belongs_to_group = True
        if belongs_to_group:
            current_group.append(box)
            min_gx = min(b[0] for b in current_group)
            max_gx = max(b[0] + b[2] for b in current_group)
            min_gy = min(b[1] for b in current_group)
            max_gy = max(b[1] + b[3] for b in current_group)
            current_group_bbox = (min_gx, min_gy, max_gx, max_gy)
        else:
            if current_group:
                min_x = min(b[0] for b in current_group)
                max_x = max(b[0] + b[2] for b in current_group)
                min_y = min(b[1] for b in current_group)
                max_y = max(b[1] + b[3] for b in current_group)
                pad_x = 10
                pad_y = 10
                min_x = max(0, min_x - pad_x)
                max_x = min(width, max_x + pad_x)
                min_y = max(0, min_y - pad_y)
                max_y = min(height, max_y + pad_y)
                total_chars = sum(len(b[4]) for b in current_group)
                groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))
            current_group = [box]
            current_group_bbox = (x, y, x + w, y + h)
    if current_group:
        min_x = min(b[0] for b in current_group)
        max_x = max(b[0] + b[2] for b in current_group)
        min_y = min(b[1] for b in current_group)
        max_y = max(b[1] + b[3] for b in current_group)
        pad_x = 10
        pad_y = 10
        min_x = max(0, min_x - pad_x)
        max_x = min(width, max_x + pad_x)
        min_y = max(0, min_y - pad_y)
        max_y = min(height, max_y + pad_y)
        total_chars = sum(len(b[4]) for b in current_group)
        groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))
    return groups

# ============ FUNZIONI DI ESTRAZIONE MIGLIORATE ============

def extract_price(text_content):
    """Estrae il prezzo principale. Formato: 7,28 € o 7.28€"""
    patterns = [
        r'(\d+)[,\.](\d{2})\s*€',
        r'(\d+)(\d{2})\s*€',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_content)
        if match:
            if len(match.group(0).replace('€', '').replace(',', '').replace('.', '').replace(' ', '')) >= 3:
                euros = match.group(1)
                cents = match.group(2)
                return f"{euros},{cents}"
    return ""

def extract_unit_price(text_content):
    """Estrae prezzo per unità. Formato: (4,55€/kg)"""
    patterns = [
        r'\((\d+[,\.]\d+)€/(\w+)\)',
        r'(\d+[,\.]\d+)\s*€\s*/\s*(\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_content)
        if match:
            price = match.group(1).replace('.', ',')
            unit = match.group(2)
            return f"{price}€/{unit}"
    return ""

def extract_unit_measure(text_content):
    """Estrae l'unità di misura dal prezzo unitario"""
    unit_price = extract_unit_price(text_content)
    if '/' in unit_price:
        return unit_price.split('/')[-1]
    return ""

def extract_rating(text_content):
    """Estrae valutazione. Formato: 4,3"""
    rating_pattern = r'\b([0-5][,\.]\d)\b'
    match = re.search(rating_pattern, text_content)
    if match:
        return match.group(1).replace('.', ',')
    return ""

def extract_reviews_count(text_content):
    """Estrae numero recensioni. Formato: (2229)"""
    reviews_pattern = r'\((\d+)\)'
    matches = re.findall(reviews_pattern, text_content)
    if matches:
        return max(matches, key=lambda x: int(x))
    return ""

def extract_delivery_info(text_content):
    """Estrae informazioni sulla consegna"""
    text_lower = text_content.lower()
    
    if 'consegna gratuita' in text_lower or 'consegna gratis' in text_lower:
        return "Consegna GRATUITA"
    
    delivery_pattern = r'[Cc]onsegna.*?(\d+[,\.]\d+)\s*€'
    match = re.search(delivery_pattern, text_content)
    if match:
        price = match.group(1).replace('.', ',')
        return f"Consegna {price}€"
    
    if 'consegna' in text_lower:
        parts = text_content.split('onsegna')
        if len(parts) > 1:
            delivery_text = "Consegna" + parts[1].strip()
            if len(delivery_text) > 100:
                delivery_text = delivery_text[:100] + "..."
            return delivery_text
    
    return ""

def check_scelta_amazon(text_content):
    """Verifica se è presente 'Scelta Amazon'"""
    text_lower = text_content.lower()
    if 'scelta amazon' in text_lower or "scelta amazon's" in text_lower:
        return 1
    return 0

def check_sponsorizzato(text_content):
    """Verifica se il prodotto è sponsorizzato"""
    text_lower = text_content.lower()
    if 'sponsorizzat' in text_lower or 'sponsor' in text_lower:
        return "Sì"
    return "No"

def analyze_product_image(product_img, product_id, debug_dir, ricerca="", user_id=""):
    """Analizza singola immagine prodotto ed estrae tutti i dati."""
    print(f"\n  Analyzing product {product_id}...")
    
    upscaled_img = upscale_image(product_img, scale_factor=2)
    text_boxes = get_text_boxes(upscaled_img, min_confidence=30)
    full_text = " ".join([box[4] for box in text_boxes])
    print(f"    OCR text: {full_text[:200]}...")
    
    data = {
        'id_utente': user_id,
        'ricerca': ricerca,
        'ordinamento': product_id,
        'sponsorizzato': check_sponsorizzato(full_text),
        'scelta_amazon': check_scelta_amazon(full_text),
        'prezzo': extract_price(full_text),
        'prezzo_per_unita': extract_unit_price(full_text),
        'unita_di_misura': extract_unit_measure(full_text),
        'prezzo_consegna': extract_delivery_info(full_text),
        'valutazione_media': extract_rating(full_text),
        'n_recensioni': extract_reviews_count(full_text),
        'login': ''
    }
    
    debug_img = Image.fromarray(cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(debug_img)
    
    colors = {
        'prezzo': 'red',
        'valutazione': 'blue',
        'recensioni': 'green',
        'unit_price': 'orange',
        'consegna': 'purple',
        'scelta_amazon': 'gold'
    }
    
    for x, y, w, h, text in text_boxes:
        box_color = 'gray'
        label = ''
        
        if re.search(r'\d+[,\.]\d{2}\s*€', text) and '/' not in text:
            box_color = colors['prezzo']
            label = 'PREZZO'
        elif re.search(r'[0-5][,\.]\d', text):
            box_color = colors['valutazione']
            label = 'RATING'
        elif re.search(r'\(\d+\)', text):
            box_color = colors['recensioni']
            label = 'REVIEWS'
        elif '€/' in text or re.search(r'\d+[,\.]\d+€/\w+', text):
            box_color = colors['unit_price']
            label = 'UNIT'
        elif 'consegna' in text.lower():
            box_color = colors['consegna']
            label = 'DELIVERY'
        elif 'scelta amazon' in text.lower():
            box_color = colors['scelta_amazon']
            label = 'SCELTA'
        
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=2)
        if label:
            draw.text((x, y-15), label, fill=box_color)
    
    debug_path = os.path.join(debug_dir, f"product_{product_id}_debug.png")
    debug_img.save(debug_path)
    
    print(f"    ✓ Prezzo: {data['prezzo']}")
    print(f"    ✓ Prezzo/unità: {data['prezzo_per_unita']}")
    print(f"    ✓ Rating: {data['valutazione_media']}")
    print(f"    ✓ Recensioni: {data['n_recensioni']}")
    print(f"    ✓ Consegna: {data['prezzo_consegna'][:50]}...")
    print(f"    ✓ Scelta Amazon: {data['scelta_amazon']}")
    
    return data

def extract_products_from_pdf(pdf_path, output_dir="output_amazon_products", 
                              ricerca="", user_id="",
                              x_ratio=0.5, y_ratio=0.1, min_chars=400, 
                              num_products_per_row=2):
    """Funzione principale per processare PDF ed estrarre dati prodotti."""
    directory = f"{output_dir}/{pdf_path.replace('.pdf', '')}"
    init_dir(directory)
    
    debug_dir = os.path.join(f"{directory}/debug_images")
    init_dir(debug_dir)

    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {num_pages}")
    print(f"Ricerca: {ricerca}")
    print(f"Products per row: {num_products_per_row}")
    print(f"{'='*60}")
    doc.close()

    all_products_data = []
    total_products_found = 0
    
    for page_num in range(num_pages):
        print(f"\n--- Page {page_num + 1}/{num_pages} ---")
        
        img = pdf_to_image(pdf_path, page_num)
        if img is None:
            continue
        
        x_threshold, y_threshold = calculate_dynamic_thresholds(img.shape, x_ratio, y_ratio)
        print(f"Thresholds - X: {x_threshold}, Y: {y_threshold}")
        
        text_boxes = get_text_boxes(img)
        image_boxes = detect_colored_regions(img, sat_thresh=40, area_thresh=30000)

        rows = assign_text_to_images(text_boxes, image_boxes, row_gap=50)
        grouped_boxes_with_chars = []
        for row in rows:
            row_groups = group_text_boxes_proximity(img.shape, row, x_threshold, y_threshold)
            grouped_boxes_with_chars.extend(row_groups)

        filtered_product_rows = [box for box in grouped_boxes_with_chars if box[4] >= min_chars]
        print(f"Found {len(filtered_product_rows)} product rows")

        for row_idx, (row_x, row_y, row_w, row_h, char_count) in enumerate(filtered_product_rows):
            row_img = img[row_y:row_y+row_h, row_x:row_x+row_w]
            
            text_boxes_in_row = []
            for text_box in text_boxes:
                tx, ty, tw, th, text = text_box
                if (row_x <= tx <= row_x + row_w and row_y <= ty <= row_y + row_h):
                    rel_x = tx - row_x
                    rel_y = ty - row_y
                    text_boxes_in_row.append((rel_x, rel_y, tw, th, text))
            
            # Use num_products_per_row parameter
            product_columns = detect_product_columns_in_row(
                row_img, text_boxes_in_row, 
                num_products=num_products_per_row
            )
            
            for col_idx, (col_x, col_y, col_w, col_h) in enumerate(product_columns):
                product_img = row_img[col_y:col_y+col_h, col_x:col_x+col_w]
                
                # Count characters in this product column
                product_text_boxes = [b for b in text_boxes_in_row 
                                     if col_x <= b[0] <= col_x + col_w]
                product_char_count = sum(len(b[4]) for b in product_text_boxes)
                
                print(f"  Product column {col_idx}: {product_char_count} chars")
                
                # Skip if below minimum character threshold
                if product_char_count < min_chars:
                    print(f"    ⚠️  Skipped (below {min_chars} chars threshold)")
                    continue
                
                total_products_found += 1
                
                product_path = os.path.join(directory, f"product_{total_products_found}.png")
                cv2.imwrite(product_path, product_img)
                
                product_data = analyze_product_image(
                    product_img, total_products_found, debug_dir, 
                    ricerca=ricerca, user_id=user_id
                )
                product_data['product_image_path'] = product_path
                product_data['page'] = page_num + 1
                product_data['row'] = row_idx
                product_data['column'] = col_idx
                
                all_products_data.append(product_data)

    if all_products_data:
        df = pd.DataFrame(all_products_data)
        
        column_order = [
            'id_utente', 'ricerca', 'ordinamento', 'sponsorizzato', 'scelta_amazon',
            'prezzo', 'prezzo_per_unita', 'unita_di_misura', 'prezzo_consegna',
            'valutazione_media', 'n_recensioni', 'login',
            'product_image_path', 'page', 'row', 'column'
        ]
        df = df.reindex(columns=column_order)
        
        excel_path = os.path.join(directory, f"products_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"\n{'='*60}")
        print(f"✓✓✓ EXTRACTION COMPLETE ✓✓✓")
        print(f"{'='*60}")
        print(f"Total products: {total_products_found}")
        print(f"Excel file: {excel_path}")
        print(f"Images: {directory}")
        print(f"Debug: {debug_dir}")
        print(f"{'='*60}")
        
        print(f"\n=== SAMPLE DATA (first 2 products) ===")
        for i, product in enumerate(all_products_data[:2]):
            print(f"\nProduct {i+1}:")
            for key, value in product.items():
                if key not in ['product_image_path', 'page', 'row', 'column']:
                    print(f"  {key}: {value}")
        
        return df, excel_path
    else:
        print("\n❌ No products found!")
        return None, None

# ESECUZIONE
if __name__ == "__main__":
    pdfs_config = [
        {"file": "candeggina.pdf", "ricerca": "amazon", "user_id": "user"}
    ]
    
    # Parametri
    x_ratio = 0.9  # proporzione orizzontale di una riga di prodotti
    y_ratio = 0.5  # proporzione verticale di una riga di prodotti
    min_char_count = 200  # Minimo caratteri per gruppo prodotto
    num_products_per_row = 2  # Numero di prodotti attesi per riga
    
    for pdf_config in pdfs_config:
        pdf_file = pdf_config["file"]
        if os.path.exists(pdf_file):
            extract_products_from_pdf(
                pdf_file, 
                ricerca=pdf_config["ricerca"],
                user_id=pdf_config["user_id"],
                x_ratio=x_ratio, 
                y_ratio=y_ratio, 
                min_chars=min_char_count,
                num_products_per_row=num_products_per_row
            )
        else:
            print(f"\n⚠️  File not found: {pdf_file}")
    
    print("\n" + "="*60)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*60)