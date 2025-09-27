import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os, shutil
import pandas as pd
import re
from datetime import datetime

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
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    
    # Apply threshold to get better contrast
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def pdf_to_image(pdf_path, page_num, dpi=300):
    """Converts a specific page of a PDF to an OpenCV image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # PyMuPDF uses 72 DPI as base
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
    image_boxes = sorted(image_boxes, key=lambda b: b[1])  # top→bottom
    text_boxes = sorted(text_boxes, key=lambda b: b[1])    # top→bottom

    for i, (x, y, w, h) in enumerate(image_boxes):
        row_top = y + h
        if i + 1 < len(image_boxes):
            row_bottom = image_boxes[i+1][1]
        else:
            row_bottom = float("inf")

        row_text = [
            b for b in text_boxes
            if row_top - row_gap <= (b[1] + b[3]//2) <= row_bottom
        ]
        if row_text:
            rows.append(row_text)
    return rows

def detect_product_columns_in_row(row_img, text_boxes_in_row, min_column_width=200, gap_threshold=50):
    """Detects individual product columns within a product row."""
    if not text_boxes_in_row:
        return [(0, 0, row_img.shape[1], row_img.shape[0])]
    
    x_positions = []
    for box in text_boxes_in_row:
        x, y, w, h, text = box
        x_positions.extend([x, x + w])
    
    x_positions = sorted(set(x_positions))
    
    gaps = []
    for i in range(len(x_positions) - 1):
        gap_start = x_positions[i]
        gap_end = x_positions[i + 1]
        gap_width = gap_end - gap_start
        
        has_text_in_gap = any(
            box[0] < gap_end and box[0] + box[2] > gap_start
            for box in text_boxes_in_row
        )
        
        if not has_text_in_gap and gap_width >= gap_threshold:
            gaps.append((gap_start, gap_end, gap_width))
    
    columns = []
    row_height = row_img.shape[0]
    row_width = row_img.shape[1]
    
    if not gaps:
        columns.append((0, 0, row_width, row_height))
    else:
        gaps = sorted(gaps, key=lambda g: g[0])
        prev_end = 0
        for gap_start, gap_end, gap_width in gaps:
            if gap_start - prev_end >= min_column_width:
                columns.append((prev_end, 0, gap_start - prev_end, row_height))
            prev_end = gap_end
        
        if row_width - prev_end >= min_column_width:
            columns.append((prev_end, 0, row_width - prev_end, row_height))
    
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

def price_regex(text, regex):
    match = re.search(regex, text)
    if match:
        euros = match.group(1)
        cents = match.group(2)
        return f"{euros},{cents}€"
    return ""
def extract_price(text_content):
    """Extract price from text content. Format: XX89 € or XXX89 €"""
    # Pattern for price: 1-3 digits + 2 digits + € symbol
    price_pattern = r'(\d{1,3})(\d{2})\s*€'
    alt_price_pattern = r'(\d{1,3})(\d{2})\s*€'
    price = price_regex(text_content, price_pattern) 
    return price_regex(text_content, alt_price_pattern) if price == "" else  price

def extract_unit_price(text_content):
    """Extract unit price from text content. Format: (X,XX€/kg)"""
    unit_pattern = r'\(([0-9,]+€/[a-zA-Z]+)\)'
    match = re.search(unit_pattern, text_content)
    if match:
        return match.group(1)
    return ""

def extract_rating(text_content):
    """Extract rating from text content. Format: X,X"""
    rating_pattern = r'\b([0-5],[0-9])\b'
    match = re.search(rating_pattern, text_content)
    if match:
        return match.group(1)
    return ""

def extract_reviews_count(text_content):
    """Extract number of reviews from text content. Format: (XXXX)"""
    # Look for numbers in parentheses, typically in blue text
    reviews_pattern = r'\((\d+)\)'
    matches = re.findall(reviews_pattern, text_content)
    # Return the largest number found in parentheses (likely reviews)
    if matches:
        return max(matches, key=lambda x: int(x))
    return ""

def extract_delivery_info(text_content):
    """Extract delivery information starting after 'Consegna'"""
    # Split by "Consegna" and take everything after it
    parts = text_content.split("Consegna")
    if len(parts) > 1:
        delivery_text = "Consegna" + parts[1].strip()
        # Clean up and limit length
        return delivery_text[:200]  # Limit to 200 characters
    return ""

def analyze_product_image(product_img, product_id, debug_dir):
    """Analyze individual product image and extract all data fields."""
    print(f"\n  Analyzing product {product_id}...")
    
    # Upscale image for better OCR
    upscaled_img = upscale_image(product_img, scale_factor=2)
    # processed_img = preprocess_for_ocr(upscaled_img)
    
    # Get all text with OCR
    text_boxes = get_text_boxes(upscaled_img, min_confidence=30)
    
    # Combine all text for content analysis
    full_text = " ".join([box[4] for box in text_boxes])
    print(f"    Full OCR text: {full_text}")
    
    # Create debug image with bounding boxes
    debug_img = Image.fromarray(cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(debug_img)
    
    # Extract data fields
    data = {
        'id_utente': '',
        'ricerca': '',  # Empty for now
        'ordinamento': product_id,  # Use product_id as ordering
        'sponsorizzato': '',  # Empty for now
        'scelta_amazon': '',  # Empty for now
        'prezzo': extract_price(full_text),
        'prezzo_per_unita': extract_unit_price(full_text),
        'unita_misura': extract_unit_price(full_text).split('/')[-1] if '/' in extract_unit_price(full_text) else '',
        'prezzo_consegna': extract_delivery_info(full_text),
        'valutazione_media': extract_rating(full_text),
        'n_recensioni': extract_reviews_count(full_text),
        'login': ''  # Empty for now
    }
    
    # Draw bounding boxes with color coding
    colors = {
        'prezzo': 'red',
        'valutazione_media': 'blue',
        'n_recensioni': 'green',
        'prezzo_per_unita': 'orange',
        'prezzo_consegna': 'purple'
    }
    
    # Highlight relevant text boxes
    for i, (x, y, w, h, text) in enumerate(text_boxes):
        # Determine what type of data this text might be
        box_color = 'gray'  # default
        
        if re.search(r'\d{1,3}\d{2}\s*€', text):  # Price pattern
            box_color = colors['prezzo']
            draw.text((x, y-15), 'PREZZO', fill=box_color)
        elif re.search(r'[0-5],[0-9]', text):  # Rating pattern
            box_color = colors['valutazione_media']
            draw.text((x, y-15), 'RATING', fill=box_color)
        elif re.search(r'\(\d+\)', text):  # Reviews pattern
            box_color = colors['n_recensioni']
            draw.text((x, y-15), 'REVIEWS', fill=box_color)
        elif '€/' in text:  # Unit price pattern
            box_color = colors['prezzo_per_unita']
            draw.text((x, y-15), 'UNIT_PRICE', fill=box_color)
        elif 'consegna' in text.lower():  # Delivery pattern
            box_color = colors['prezzo_consegna']
            draw.text((x, y-15), 'DELIVERY', fill=box_color)
        
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=2)
    
    # Save debug image
    debug_path = os.path.join(debug_dir, f"product_{product_id}_debug.png")
    debug_img.save(debug_path)
    
    print(f"    Extracted data: {data}")
    print(f"    Debug image saved: {debug_path}")
    
    return data

def extract_products_from_pdf(pdf_path, output_dir="output_products_data", x_threshold=400, y_threshold=100, min_chars=400):
    """Main function to process PDF and extract all product data."""
    directory = f"{output_dir}/{pdf_path.replace('.pdf', '')}"
    init_dir(directory)
    
    debug_dir = os.path.join(f"{directory}/debug_images")
    init_dir(debug_dir)

    # Open PDF
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"Processing PDF with {num_pages} page(s).")
    doc.close()

    all_products_data = []
    total_products_found = 0
    
    for page_num in range(num_pages):
        print(f"\n--- Processing Page {page_num + 1} ---")
        
        # Convert PDF page to image
        img = pdf_to_image(pdf_path, page_num)
        if img is None:
            continue

        # Get text boxes and detect regions
        text_boxes = get_text_boxes(img)
        image_boxes = detect_colored_regions(img, sat_thresh=40, area_thresh=30000)

        # Group into rows and then products
        rows = assign_text_to_images(text_boxes, image_boxes, row_gap=50)
        grouped_boxes_with_chars = []
        for row in rows:
            row_groups = group_text_boxes_proximity(img.shape, row, x_threshold, y_threshold)
            grouped_boxes_with_chars.extend(row_groups)

        # Filter by character count
        filtered_product_rows = [box for box in grouped_boxes_with_chars if box[4] >= min_chars]
        print(f"Found {len(filtered_product_rows)} product rows on page {page_num + 1}")

        # Process each product row
        for row_idx, (row_x, row_y, row_w, row_h, char_count) in enumerate(filtered_product_rows):
            # Extract row image
            row_img = img[row_y:row_y+row_h, row_x:row_x+row_w]
            
            # Get text boxes in row
            text_boxes_in_row = []
            for text_box in text_boxes:
                tx, ty, tw, th, text = text_box
                if (row_x <= tx <= row_x + row_w and row_y <= ty <= row_y + row_h):
                    rel_x = tx - row_x
                    rel_y = ty - row_y
                    text_boxes_in_row.append((rel_x, rel_y, tw, th, text))
            
            # Detect individual products in row
            product_columns = detect_product_columns_in_row(
                row_img, text_boxes_in_row, 
                min_column_width=150, gap_threshold=50
            )
            
            # Process each individual product
            for col_idx, (col_x, col_y, col_w, col_h) in enumerate(product_columns):
                total_products_found += 1
                
                # Extract product image
                product_img = row_img[col_y:col_y+col_h, col_x:col_x+col_w]
                
                # Save product image
                product_path = os.path.join(directory, f"product_{total_products_found}.png")
                cv2.imwrite(product_path, product_img)
                
                # Analyze product and extract data
                product_data = analyze_product_image(product_img, total_products_found, debug_dir)
                product_data['product_image_path'] = product_path
                product_data['page'] = page_num + 1
                product_data['row'] = row_idx
                product_data['column'] = col_idx
                
                all_products_data.append(product_data)

    # Create Excel file
    if all_products_data:
        df = pd.DataFrame(all_products_data)
        # Reorder columns as requested
        column_order = [
            'id_utente', 'ricerca', 'ordinamento', 'sponsorizzato', 'scelta_amazon',
            'prezzo', 'prezzo_per_unita', 'unita_misura', 'prezzo_consegna',
            'valutazione_media', 'n_recensioni', 'login',
            'product_image_path', 'page', 'row', 'column'
        ]
        df = df.reindex(columns=column_order)
        
        excel_path = os.path.join(directory, f"products_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Total products extracted: {total_products_found}")
        print(f"Excel file created: {excel_path}")
        print(f"Product images saved in: {directory}")
        print(f"Debug images saved in: {debug_dir}")
        
        # Print sample of extracted data
        print(f"\n=== SAMPLE DATA ===")
        for i, product in enumerate(all_products_data[:3]):  # Show first 3 products
            print(f"Product {i+1}:")
            for key, value in product.items():
                if key not in ['product_image_path', 'page', 'row', 'column']:
                    print(f"  {key}: {value}")
            print()
    else:
        print("No products found!")

# Run the extraction
if __name__ == "__main__":
    pdfs = ["page_3.pdf", "page_9.pdf", "ricerca_animali.pdf"]
    # Parameters
    x_threshold = 400  # Horizontal grouping threshold
    y_threshold = 100  # Vertical grouping threshold
    min_char_count = 400  # Minimum characters for a product group
    for pdf in pdfs:    
        extract_products_from_pdf(pdf, x_threshold=x_threshold, y_threshold=y_threshold, min_chars=min_char_count)