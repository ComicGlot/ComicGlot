import logging
import cv2
import os
import numpy as np
from typing import List, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

def crop_image(image: Union[str, cv2.Mat], bbox: Tuple[int, int, int, int]) -> cv2.Mat:
    """
    Crop the image based on the bounding box, with fallback to adjust coordinates
    if they are out of bounds or contact the edges of the image.

    Args:
        image (Union[str, cv2.Mat]): Path to the image file or cv2 image object.
        bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        np.ndarray: Cropped image or None if the crop is invalid.
    """
    if isinstance(image, str):  # If image is a path, load it
        image = load_image(image)

    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(x_min + 1, min(w, x_max))  # Ensure x_max > x_min
    y_max = max(y_min + 1, min(h, y_max))  # Ensure y_max > y_min

    if x_min >= x_max or y_min >= y_max:
        logging.warning(f"Adjusted bounding box is invalid: {bbox} -> {(x_min, y_min, x_max, y_max)}")
        return None

    cropped = image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        logging.warning(f"Cropped image is empty for bbox: {bbox}")
        return None

    return cropped

def load_image(image_path: str) -> cv2.Mat:
    """
    Load the image to cv2 format.
    Args:
        image_path (str): Path to the image file.

    Returns:
        cv2.Mat: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

# WRITE TYPES HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \/
def draw_bounding_box_with_text(image: cv2.Mat, bubble_box, class_name, confidence, ocr_results) -> cv2.Mat:
    """
    Draws a bubble bounding box and string boxes with annotations on an image.
    
    Args:
        image (cv2.Mat): The image to annotate.
        bubble_box (tuple): Bubble bounding box as (x_min, y_min, x_max, y_max).
        class_name (str): Class name of the detected object.
        confidence (float): Confidence score.
        ocr_results (list): List of OCR results, where each entry is (bbox, text).

    Returns:
        cv2.Mat: Annotated image.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    x_min, y_min, x_max, y_max = map(int, bubble_box)

    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=4)
    bubble_label = f"{class_name} ({confidence:.2f})"

    try:
        FONT_PATH = os.path.abspath(os.getenv("FONT_PATH", "./data/fonts/IBMPlexSansKR-Medium.ttf"))
        font = ImageFont.truetype(FONT_PATH, 24)
    except IOError:
        font = ImageFont.load_default()

    bubble_label_bbox = draw.textbbox((x_min, y_min), bubble_label, font=font)
    bubble_label_width, bubble_label_height = bubble_label_bbox[2] - bubble_label_bbox[0], bubble_label_bbox[3] - bubble_label_bbox[1] + 5 # Correcting text spot due to font!
    bubble_label_position = (x_min, y_min - bubble_label_height - 2 if y_min - bubble_label_height > 0 else y_min + 2)

    # Draw the label background
    draw.rectangle(
        [bubble_label_position, (bubble_label_position[0] + bubble_label_width, bubble_label_position[1] + bubble_label_height)],
        fill="yellow"
    )
    # Draw the bubble label text. Correcting text spot due to font!
    draw.text((bubble_label_position[0], bubble_label_position[1] - 5), bubble_label, fill="black", font=font)

    # Loop through OCR results and draw text boxes
    for string_box, string_text in ocr_results:
        string_x_min, string_y_min, string_x_max, string_y_max = map(int, string_box)
        # Adjust bubble box with offsets
        string_x_min += x_min
        string_y_min += y_min
        string_x_max += x_min
        string_y_max += y_min

        # Draw the string bounding box
        draw.rectangle([string_x_min, string_y_min, string_x_max, string_y_max], outline="blue", width=4)

        text_label = string_text.strip()
        text_label_bbox = draw.textbbox((string_x_min, string_y_min), text_label, font=font)
        text_label_width, text_label_height = text_label_bbox[2] - text_label_bbox[0], text_label_bbox[3] - text_label_bbox[1] + 5 # Correcting text spot due to font!
        text_label_position = (
            string_x_min, 
            string_y_min - text_label_height - 2 if string_y_min - text_label_height > 0 else string_y_min + 2
        )

        # Draw the label background for text
        draw.rectangle(
            [text_label_position, (text_label_position[0] + text_label_width, text_label_position[1] + text_label_height)],
            fill="lightgreen"
        )
        # Draw the text
        draw.text((text_label_position[0], text_label_position[1] - 5), text_label, fill="black", font=font)

    # Convert back to OpenCV format (numpy array)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_next_run_folder(base_output_folder: str) -> str:
    """
    Find the smallest available "run{number}" folder in the output directory.
    
    Args:
        base_output_folder (str): The base output directory.
    
    Returns:
        str: Path to the new "run{number}" folder.
    """
    run_number = 1
    while True:
        run_folder = os.path.join(base_output_folder, f"run{run_number}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        run_number += 1

def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode = 'ltr'):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append([box[1], min_x, max_x, min_y, max_y, height, 0.5*(min_y+max_y), 0]) # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7]==0]) > 0:
        box_group0 = [box for box in box_group if box[7]==0] # group0 = non-group
        # new group
        if len([box for box in box_group if box[7]==current_group]) == 0:
            box_group0[0][7] = current_group # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7]==current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths*mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths*mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths*mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths*mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx<=box[1]<=max_gx) or (min_gx<=box[2]<=max_gx)
                same_vertical_level = (min_gy<=box[3]<=max_gy) or (min_gy<=box[4]<=max_gy)
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box==False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7]==i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ''
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6]<highest+0.4*mean_height]
            # get the far left
            if mode == 'ltr':
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left: best_box = box
            elif mode == 'rtl':
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right: best_box = box
            text += ' '+best_box[0]
            current_box_group.remove(best_box)

        result.append([ [[min_gx,min_gy],[max_gx,min_gy],[max_gx,max_gy],[min_gx,max_gy]], text[1:]])

    return result


def apply_separator_mask(cropped_block: np.ndarray, separator_line: np.ndarray) -> np.ndarray:
    """
    Creates a mask from the separator line and applies it to the cropped block.
    
    Args:
        cropped_block (np.ndarray): Cropped image block.
        separator_line (np.ndarray): Two points representing the separator line.

    Returns:
        np.ndarray: Cropped block with one side of the separator removed.
    """

    (x1, y1), (x2, y2) = separator_line.reshape(2, 2)
    h, w = cropped_block.shape[:2]

    # Compute line equation coefficients: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Find the center of the cropped block
    cx, cy = w // 2, h // 2

    # Determine which side to keep
    center_side = A * cx + B * cy + C
    keep_white = center_side > 0  # If True, we fill the opposite side

    mask = np.zeros((h, w), dtype=np.uint8)

    if keep_white:
        fill_side = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
    else:
        fill_side = np.array([[0, 0], [w, 0], [x2, y2], [x1, y1], [w, h], [0, h]], dtype=np.int32)

    cv2.fillPoly(mask, [fill_side], 255)
    result = cv2.bitwise_and(cropped_block, cropped_block, mask=mask)

    return result