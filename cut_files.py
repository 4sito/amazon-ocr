import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import os, shutil

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

def pdf_to_image(pdf_path, page_num, dpi=300):
    """Converts a specific page of a PDF to an OpenCV image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # PyMuPDF uses 72 DPI as base
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    doc.close() # Close doc after extracting the page image
    return img

def get_text_boxes(img, min_confidence=30):
    """Gets bounding boxes for all text elements using OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Optional: Apply thresholding for potentially better OCR results
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pil_img = Image.fromarray(thresh)

    # Use pytesseract to get data including bounding boxes
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='ita')

    boxes = []
    n_boxes = len(data['level']) # Use 'level' instead of 'conf' for length check
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        if conf > min_confidence:
            text = data['text'][i].strip()
            if text: # Only consider non-empty text
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                boxes.append((x, y, w, h, text))
                # print(f"OCR Text: '{text}' at ({x}, {y}, {w}, {h}), Conf: {conf}") # Debug print
    return boxes

def assign_text_to_images(text_boxes, image_boxes, row_gap=50):
    """
    Assigns OCR text boxes to rows defined by product images.
    Each image anchors a row: all text below it until the next image belongs to that row.
    """
    rows = []
    image_boxes = sorted(image_boxes, key=lambda b: b[1])  # top→bottom
    text_boxes = sorted(text_boxes, key=lambda b: b[1])    # top→bottom

    for i, (x, y, w, h) in enumerate(image_boxes):
        row_top = y + h
        if i + 1 < len(image_boxes):
            row_bottom = image_boxes[i+1][1]   # start of next image
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
    """
    Detects individual product columns within a product row.
    
    Args:
        row_img: The image of the product row
        text_boxes_in_row: List of text boxes within this row (relative to row coordinates)
        min_column_width: Minimum width for a column to be considered valid
        gap_threshold: Minimum gap between columns
    
    Returns:
        List of column bounding boxes (x, y, w, h) relative to the row image
    """
    if not text_boxes_in_row:
        # If no text boxes, return the entire row as one column
        return [(0, 0, row_img.shape[1], row_img.shape[0])]
    
    # Get horizontal positions of all text elements
    x_positions = []
    for box in text_boxes_in_row:
        x, y, w, h, text = box
        x_positions.extend([x, x + w])  # Add both left and right edges
    
    x_positions = sorted(set(x_positions))  # Remove duplicates and sort
    
    # Find gaps between text elements
    gaps = []
    for i in range(len(x_positions) - 1):
        gap_start = x_positions[i]
        gap_end = x_positions[i + 1]
        gap_width = gap_end - gap_start
        
        # Check if this gap has no text overlapping
        has_text_in_gap = any(
            box[0] < gap_end and box[0] + box[2] > gap_start
            for box in text_boxes_in_row
        )
        
        if not has_text_in_gap and gap_width >= gap_threshold:
            gaps.append((gap_start, gap_end, gap_width))
    
    # Create columns based on gaps
    columns = []
    row_height = row_img.shape[0]
    row_width = row_img.shape[1]
    
    if not gaps:
        # No significant gaps found, treat as single column
        columns.append((0, 0, row_width, row_height))
    else:
        # Sort gaps by position
        gaps = sorted(gaps, key=lambda g: g[0])
        
        # Create columns between gaps
        prev_end = 0
        for gap_start, gap_end, gap_width in gaps:
            if gap_start - prev_end >= min_column_width:
                columns.append((prev_end, 0, gap_start - prev_end, row_height))
            prev_end = gap_end
        
        # Add the last column
        if row_width - prev_end >= min_column_width:
            columns.append((prev_end, 0, row_width - prev_end, row_height))
    
    return columns

def detect_colored_regions(img, sat_thresh=40, area_thresh=5000):
    """
    Detects colored regions (likely product images) in the page.
    sat_thresh: minimum saturation for a pixel to be considered colored
    area_thresh: minimum contour area to keep (to ignore noise)
    Returns a list of bounding boxes (x, y, w, h).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract saturation channel
    s = hsv[:, :, 1]

    # Threshold for high saturation
    _, mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)

    # Find contours of high-saturation regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= area_thresh:
            boxes.append((x, y, w, h))
    return boxes

def group_text_boxes_proximity(img_shape, text_boxes, x_threshold=100, y_threshold=100):
    """
    Groups text boxes based on their spatial proximity within a rectangular area.
    A new box is added to the current group if its center falls within the
    current group's bounding box expanded by x_threshold and y_threshold.
    """
    if not text_boxes:
        print("No text boxes found, cannot determine product areas.")
        return []

    # Sort boxes primarily by top coordinate (y), then by left (x)
    # This helps in processing top-to-bottom, left-to-right
    sorted_boxes = sorted(text_boxes, key=lambda b: (b[1], b[0]))

    groups = []
    current_group = []
    current_group_bbox = None # (min_x, min_y, max_x, max_y)

    height, width = img_shape[:2]

    for box in sorted_boxes:
        x, y, w, h, text = box
        box_center_x = x + w // 2
        box_center_y = y + h // 2

        belongs_to_group = False
        if current_group and current_group_bbox:
            min_gx, min_gy, max_gx, max_gy = current_group_bbox
            # Define the expanded group area
            expanded_min_x = min_gx - x_threshold
            expanded_max_x = max_gx + x_threshold
            expanded_min_y = min_gy - y_threshold
            expanded_max_y = max_gy + y_threshold

            # Check if the center of the new box is within the expanded group area
            if (expanded_min_x <= box_center_x <= expanded_max_x and
                expanded_min_y <= box_center_y <= expanded_max_y):
                 belongs_to_group = True

        if belongs_to_group:
            # Add to current group
            current_group.append(box)
            # Update the group's bounding box
            min_gx = min(b[0] for b in current_group)
            max_gx = max(b[0] + b[2] for b in current_group)
            min_gy = min(b[1] for b in current_group)
            max_gy = max(b[1] + b[3] for b in current_group)
            current_group_bbox = (min_gx, min_gy, max_gx, max_gy)
        else:
            # Finalize the previous group if it exists
            if current_group:
                # Calculate the final bounding box for the group with padding
                min_x = min(b[0] for b in current_group)
                max_x = max(b[0] + b[2] for b in current_group)
                min_y = min(b[1] for b in current_group)
                max_y = max(b[1] + b[3] for b in current_group)
                # Add padding
                pad_x = 10
                pad_y = 10
                min_x = max(0, min_x - pad_x)
                max_x = min(width, max_x + pad_x)
                min_y = max(0, min_y - pad_y)
                max_y = min(height, max_y + pad_y)
                # Calculate total character count in the group
                total_chars = sum(len(b[4]) for b in current_group) # Sum length of text in each box
                groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))

            # Start a new group
            current_group = [box]
            current_group_bbox = (x, y, x + w, y + h) # Initialize bbox with the first box

    # Finalize the last group
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
        total_chars = sum(len(b[4]) for b in current_group) # Sum length of text in each box
        groups.append((min_x, min_y, max_x - min_x, max_y - min_y, total_chars))

    return groups

def extract_products_from_pdf(pdf_path, output_dir="output_products_individual", debug_path_prefix="debug_output_page_", x_threshold=200, y_threshold=150, min_chars=100):
    """Main function to process all pages of the PDF and extract individual products."""
    directory = f"{output_dir}-{x_threshold}-{y_threshold}/{pdf_path}"
    os.makedirs(directory, exist_ok=True)

    clear_directory(directory)

    # Open the PDF once to get the number of pages
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"Processing PDF with {num_pages} page(s).")
    doc.close() # Close immediately after getting page count

    total_products_found = 0
    for page_num in range(num_pages):
        print(f"\n--- Processing Page {page_num + 1} ---")
        # Step 1: Convert PDF page to image
        print("Converting page to image...")
        img = pdf_to_image(pdf_path, page_num)
        if img is None:
            print(f"Error: Could not load image from page {page_num + 1}. Skipping page.")
            continue # Skip to the next page if loading fails

        # Step 2: Get all text bounding boxes
        print("Getting text boxes using OCR...")
        text_boxes = get_text_boxes(img)

        print("Detecting colored regions...")
        image_boxes = detect_colored_regions(img, sat_thresh=40, area_thresh=30000)

        # Step 3: First cluster into rows
        rows = assign_text_to_images(text_boxes, image_boxes, row_gap=50)
        grouped_boxes_with_chars = []
        for row in rows:
            row_groups = group_text_boxes_proximity(img.shape, row, x_threshold, y_threshold)
            grouped_boxes_with_chars.extend(row_groups)

        # Filter groups based on character count
        print(f"Filtering groups with at least {min_chars} characters...")
        filtered_product_rows = [box for box in grouped_boxes_with_chars if box[4] >= min_chars]

        print(f"Found {len(grouped_boxes_with_chars)} groups before filtering, {len(filtered_product_rows)} after filtering on page {page_num + 1}.")

        # Step 4: Process each product row to find individual products
        page_products = []
        debug_img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(debug_img_pil)

        # Draw image boxes (blue)
        for (x, y, w, h) in image_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="blue", width=3)

        # Draw rows (green)
        for i, row in enumerate(rows):
            if not row:
                continue
            min_x = min(b[0] for b in row)
            max_x = max(b[0] + b[2] for b in row)
            min_y = min(b[1] for b in row)
            max_y = max(b[1] + b[3] for b in row)
            draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=2)

        for row_idx, (row_x, row_y, row_w, row_h, char_count) in enumerate(filtered_product_rows):
            print(f"\n  Processing product row {row_idx} on page {page_num + 1}...")
            
            # Extract the row image
            row_img = img[row_y:row_y+row_h, row_x:row_x+row_w]
            
            # Get text boxes that fall within this row (convert coordinates to row-relative)
            text_boxes_in_row = []
            for text_box in text_boxes:
                tx, ty, tw, th, text = text_box
                # Check if text box is within the row boundaries
                if (row_x <= tx <= row_x + row_w and row_y <= ty <= row_y + row_h):
                    # Convert to row-relative coordinates
                    rel_x = tx - row_x
                    rel_y = ty - row_y
                    text_boxes_in_row.append((rel_x, rel_y, tw, th, text))
            
            # Detect individual product columns within this row
            product_columns = detect_product_columns_in_row(
                row_img, 
                text_boxes_in_row, 
                min_column_width=150,  # Adjust as needed
                gap_threshold=50       # Adjust as needed
            )
            
            print(f"    Found {len(product_columns)} products in row {row_idx}")
            
            # Draw row boundary (red)
            draw.rectangle([row_x, row_y, row_x + row_w, row_y + row_h], outline="red", width=3)
            draw.text((row_x + 5, row_y + 5), f"Row {row_idx}", fill="red")
            
            # Extract and save individual products
            for col_idx, (col_x, col_y, col_w, col_h) in enumerate(product_columns):
                # Convert column coordinates back to original image coordinates
                abs_col_x = row_x + col_x
                abs_col_y = row_y + col_y
                
                # Draw product boundary (yellow)
                draw.rectangle([abs_col_x, abs_col_y, abs_col_x + col_w, abs_col_y + col_h], 
                             outline="yellow", width=2)
                draw.text((abs_col_x + 5, abs_col_y + 20), f"P{total_products_found}", fill="yellow")
                
                # Extract product image
                product_img = row_img[col_y:col_y+col_h, col_x:col_x+col_w]
                
                # Save individual product
                output_path = os.path.join(directory, f"product_page{page_num + 1}_row{row_idx}_col{col_idx}.png")
                cv2.imwrite(output_path, product_img)
                
                page_products.append({
                    'page': page_num + 1,
                    'row': row_idx,
                    'column': col_idx,
                    'bbox': (abs_col_x, abs_col_y, col_w, col_h),
                    'path': output_path
                })
                
                print(f"      Saved product {total_products_found} to {output_path}")
                total_products_found += 1

        # Save debug image
        debug_path = f"{directory}/{debug_path_prefix}{page_num + 1}.png"
        debug_img_pil.save(debug_path)
        print(f"Saved debug image: {debug_path}")

    print(f"\nAll pages processed. Total individual products extracted: {total_products_found}")
    print(f"Check '{directory}' for individual product images and debug images.")

# --- Run the extraction ---

input_pdf_path = "page_3.pdf" # "amazon.pdf" / "ricerca_animali.pdf"

# Thresholds for grouping text boxes (pixels)
# x_threshold: How far horizontally a text box can be from the group's bounding box
# y_threshold: How far vertically a text box can be from the group's bounding box
x_threshold = 400  # Adjust this value as needed (e.g., 150, 200, 300)
y_threshold = 100  # Adjust this value as needed (e.g., 100, 150, 200)

# Minimum character count for a group to be considered a product
min_char_count = 400 # Adjust this value as needed (e.g., 100, 150, 200, 300)

x_steps = [400] #[250, 300, 350]
y_steps = [200]
for step in x_steps:
    # --- Run the extraction ---
    extract_products_from_pdf(input_pdf_path, x_threshold=step, y_threshold=y_threshold, min_chars=min_char_count)