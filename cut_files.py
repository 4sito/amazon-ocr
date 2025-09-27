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

# def cluster_by_rows(text_boxes, row_gap=50):
#     """
#     Groups text boxes into rows based on their vertical position.
#     row_gap: max vertical distance between centers to be considered the same row
#     """
#     if not text_boxes:
#         return []

#     # Sort by vertical position (y coordinate of center)
#     text_boxes = sorted(text_boxes, key=lambda b: b[1] + b[3]//2)

#     rows = []
#     current_row = [text_boxes[0]]
#     current_center_y = text_boxes[0][1] + text_boxes[0][3]//2

#     for box in text_boxes[1:]:
#         _, y, _, h, _ = box
#         center_y = y + h//2
#         if abs(center_y - current_center_y) <= row_gap:
#             current_row.append(box)
#             current_center_y = np.mean([b[1] + b[3]//2 for b in current_row])
#         else:
#             rows.append(current_row)
#             current_row = [box]
#             current_center_y = center_y

#     rows.append(current_row)
#     return rows

def row_overlaps_image(row_boxes, image_boxes):
    """
    Check if a row of text overlaps with any detected image box.
    """
    if not row_boxes or not image_boxes:
        return False
    # Row vertical span
    min_y = min(b[1] for b in row_boxes)
    max_y = max(b[1] + b[3] for b in row_boxes)
    for (x, y, w, h) in image_boxes:
        if y < max_y and (y + h) > min_y:  # vertical overlap
            return True
    return False


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


def extract_products_from_pdf(pdf_path, output_dir="output_products_text_group", debug_path_prefix="debug_output_page_", x_threshold=200, y_threshold=150, min_chars=100):
    """Main function to process all pages of the PDF and extract products using text clustering."""
    directory = f"{output_dir}-{x_threshold}-{y_threshold}/{pdf_path}"
    os.makedirs(directory, exist_ok=True)

    clear_directory(directory)

    # Open the PDF once to get the number of pages
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"Processing PDF with {num_pages} page(s).")
    doc.close() # Close immediately after getting page count

    total_products_rows_found = 0
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
        # rows = cluster_by_rows(text_boxes, row_gap=650)  # adjust row_gap as needed
        rows = assign_text_to_images(text_boxes, image_boxes, row_gap=50)
        grouped_boxes_with_chars = []
        for row in rows:
            row_groups = group_text_boxes_proximity(img.shape, row, x_threshold, y_threshold)
            grouped_boxes_with_chars.extend(row_groups)

        # Filter groups based on character count
        print(f"Filtering groups with at least {min_chars} characters...")
        filtered_product_boxes = [box for box in grouped_boxes_with_chars if box[4] >= min_chars] # box[4] is the character count

        print(f"Found {len(grouped_boxes_with_chars)} groups before filtering, {len(filtered_product_boxes)} after filtering on page {page_num + 1}.")

        # Step 4: Create debug image for the current page (only show filtered boxes)
        debug_path = f"{directory}/{debug_path_prefix}{page_num + 1}.png"
        print(f"Creating debug image for page {page_num + 1}: {debug_path}")
        debug_img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(debug_img_pil)

        # --- Draw product image detections (blue) ---
        for (x, y, w, h) in image_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="blue", width=3)

        # --- Draw row spans (green) ---
        for i, row in enumerate(rows):
            if not row:
                continue
            min_x = min(b[0] for b in row)
            max_x = max(b[0] + b[2] for b in row)
            min_y = min(b[1] for b in row)
            max_y = max(b[1] + b[3] for b in row)
            draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=3)
            draw.text((min_x + 5, min_y - 15), f"Row {i}", fill="green")

        # --- Draw final product groups (red) ---
        for i, (x, y, w, h, char_count) in enumerate(filtered_product_boxes):
            print(f"  Product row {total_products_rows_found + i} (Page {page_num + 1}): ({x}, {y}, {w}, {h}), Chars: {char_count}")
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            draw.text((x + 5, y + 5), f"Prod {total_products_rows_found + i} ({char_count})", fill="red")
            debug_img_pil.save(debug_path)


        # Step 5: Extract and save individual product images (only from filtered boxes) for the current page
        print(f"Extracting {len(filtered_product_boxes)} products rows from page {page_num + 1}...")
        for i, (x, y, w, h, char_count) in enumerate(filtered_product_boxes):
            product_img = img[y:y+h, x:x+w]
            # Use a unique filename including page and product number
            output_path = os.path.join(directory, f"product_row_{page_num + 1}_num{i}.png")
            cv2.imwrite(output_path, product_img)
            print(f"  Saved product row {total_products_rows_found + i} (chars: {char_count}) from page {page_num + 1} to {output_path}")

        total_products_rows_found += len(filtered_product_boxes)

    print(f"\nAll pages processed. Total products extracted: {total_products_rows_found}")
    print(f"Check '{directory}' for product row images and '{debug_path_prefix}*.png' for the debug images.")

# --- Run the extraction ---

input_pdf_path = "amazon.pdf" # "amazon.pdf" / "ricerca_animali.pdf"

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