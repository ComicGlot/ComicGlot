import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
from data_types import TextBlock

def draw_bounding_box(
    image: np.ndarray, 
    box: Tuple[int, int, int, int], 
    label: Optional[str] = None, 
    color: Tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box and optional label on an image.
    
    Args:
        image: Input image
        box: Bounding box coordinates (x_min, y_min, x_max, y_max)
        label: Optional text label to display
        color: RGB color tuple for the box and label
        thickness: Line thickness for the box
        
    Returns:
        Image with bounding box drawn
    """
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    if label:
        font_scale = 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

        cv2.rectangle(
            image, 
            (x_min, y_min - text_size[1] - 5), 
            (x_min + text_size[0], y_min), 
            color, 
            -1  # Filled rectangle
        )

        cv2.putText(
            image, 
            label, 
            (x_min, y_min - 5), 
            font, 
            font_scale, 
            (0, 0, 0),  # Black text
            1, 
            cv2.LINE_AA
        )
    
    return image

def draw_boxes_with_labels(
    image: np.ndarray,
    boxes: List[np.ndarray],
    classes: List[str],
    confidences: List[float],
    ids: Optional[List[int]] = None,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        classes: List of class names
        confidences: List of confidence scores
        ids: Optional list of box IDs
        color_map: Optional mapping of class names to colors
        line_thickness: Thickness of bounding box lines
        
    Returns:
        Image with drawn boxes and labels
    """
    from PIL import Image, ImageDraw, ImageFont

    vis_image = image.copy()

    if color_map is None:
        color_map = {
            "speech_bubble": (255, 0, 0),       # Red
            "cloud_bubble": (255, 192, 203),    # Pink
            "scream_bubble": (255, 69, 0),      # Orange-Red
            "spiked_bubble": (139, 0, 0),       # Dark Red
            "system_bubble": (0, 0, 255),       # Blue
            "think_bubble": (0, 191, 255),      # Deep Sky Blue
            "standalone_text": (0, 255, 0),     # Green
            "sfx_sound": (255, 215, 0),         # Gold
            "hexagon_bubble": (148, 0, 211),    # Dark Violet
            "square_bubble": (75, 0, 130),      # Indigo
            "wiggle_bubble": (0, 255, 255),     # Cyan
            "title_panel": (255, 165, 0),       # Orange
            "background_text": (128, 128, 128)  # Gray
        }

    pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    try:
        FONT_PATH = os.path.abspath(os.getenv("FONT_PATH", "./data/fonts/IBMPlexSansKR-Medium.ttf"))
        font = ImageFont.truetype(FONT_PATH, 24)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, class_name, confidence) in enumerate(zip(boxes, classes, confidences)):
        x_min, y_min, x_max, y_max = map(int, box)

        if class_name in color_map:
            color_rgb = color_map[class_name]
        else:
            color_rgb = (255, 0, 0)

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color_rgb, width=line_thickness)

        if ids is not None:
            label = f"{ids[i]}: {class_name} ({confidence:.2f})"
        else:
            label = f"{class_name} ({confidence:.2f})"

        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1] + 5  # Correcting text spot due to font

        label_position = (
            x_min, 
            y_min - label_height - 2 if y_min - label_height > 0 else y_min + 2
        )

        draw.rectangle(
            [
                label_position, 
                (label_position[0] + label_width, label_position[1] + label_height)
            ],
            fill=color_rgb
        )

        draw.text(
            (label_position[0], label_position[1] - 5), 
            label, 
            fill=(255, 255, 255),  # White text
            font=font
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def visualize_separator_lines(
    image: np.ndarray,
    blocks: List[TextBlock],
    separator_lines_lists: List[List[np.ndarray]],
    line_thickness: int = 2,
    line_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
) -> np.ndarray:
    """
    Visualize separator lines between overlapping text blocks.
    
    Args:
        image: Input image
        blocks: List of TextBlock objects
        separator_lines_lists: List of lists of separator lines
        line_thickness: Thickness of separator lines
        line_color: Color of separator lines
        
    Returns:
        Image with drawn separator lines
    """
    vis_image = image.copy()

    for block_idx, separator_lines in enumerate(separator_lines_lists):
        if not separator_lines:
            continue

        block = blocks[block_idx]
        x1, y1, x2, y2 = map(int, block.box)
        
        for line in separator_lines:
            (line_x1, line_y1), (line_x2, line_y2) = line.reshape(2, 2)

            abs_line_x1 = int(line_x1)
            abs_line_y1 = int(line_y1)
            abs_line_x2 = int(line_x2)
            abs_line_y2 = int(line_y2)

            cv2.line(
                vis_image,
                (abs_line_x1, abs_line_y1),
                (abs_line_x2, abs_line_y2),
                line_color,
                line_thickness
            )

            cv2.circle(vis_image, (abs_line_x1, abs_line_y1), line_thickness + 1, line_color, -1)
            cv2.circle(vis_image, (abs_line_x2, abs_line_y2), line_thickness + 1, line_color, -1)
    
    return vis_image
    