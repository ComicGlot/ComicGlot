import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
from natsort import natsorted

from comicglot.utils.file_utils import load_image
from comicglot.utils.image_utils import resize_image, create_empty_image

def segment_by_seam_carving(
    input_path: str, 
    min_height: int = 800, 
    save_crops: bool = False, 
    output_path: Optional[str] = None, 
    resize: bool = False, 
    tolerance: float = 0.9, 
    carving_step: int = 20
) -> Dict:
    """
    Segments manga/manhwa images using seam carving algorithm to find natural division points.
    
    Args:
        input_path: Path to folder containing images
        min_height: Minimum height of each segment in pixels
        save_crops: Whether to save individual crops to output_path
        output_path: Path to save crops (if None and save_crops is True, uses input_path/crops)
        resize: If True, resizes all images to the same width; otherwise uses black padding
        tolerance: Threshold for color homogeneity (0.0-1.0)
        carving_step: Window size for finding local maxima of homogeneity
    
    Returns:
        Dict containing:
            'original_path': Path to input folder
            'segments': List of image segments as numpy arrays
            'segment_boundaries': List of (start, end) positions in original image
            'composite_height': Height of the composite image
            'composite_width': Width of the composite image
    """
    logging.info(f"Starting seam carving segmentation on: {input_path}")
    
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if save_crops and output_path is None:
        output_path = os.path.join(input_path, "crops")
    
    if save_crops:
        os.makedirs(output_path, exist_ok=True)
        logging.info(f"Crops will be saved to: {output_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    files = os.listdir(input_path)
    
    for file in natsorted(files):
        file_path = os.path.join(input_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)
    
    if not image_files:
        raise ValueError(f"No image files found in: {input_path}")
    
    logging.info(f"Found {len(image_files)} image files")
    
    images = []
    max_width = 0
    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is None:
            logging.warning(f"Could not read image: {file_path}, skipping")
            continue
        images.append(img)
        max_width = max(max_width, img.shape[1])
    
    logging.info(f"Loaded {len(images)} images, max width: {max_width}")
    
    processed_images = []
    for img in images:
        h, w = img.shape[:2]
        
        if resize and w != max_width:
            # Resize image to match max width
            scale_factor = max_width / w
            new_height = int(h * scale_factor)
            resized_img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_CUBIC)
            processed_images.append(resized_img)
        elif not resize and w < max_width:
            # Pad to the widest image by repeating the right edge column. Black
            # padding made every row of a narrower image fail the homogeneity
            # test (the jump into black exceeds the 50-unit penalty), so those
            # images could never be cut. The pipeline itself calls this with
            # resize=True and is unaffected.
            padded_img = cv2.copyMakeBorder(img, 0, 0, 0, max_width - w, cv2.BORDER_REPLICATE)
            processed_images.append(padded_img)
        else:
            # Image already has max width
            processed_images.append(img)
    
    heights = [img.shape[0] for img in processed_images]
    total_height = sum(heights)
    composite_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    
    y_offset = 0
    for img in processed_images:
        h = img.shape[0]
        composite_image[y_offset:y_offset+h, :, :] = img
        y_offset += h
    
    logging.info(f"Created composite image of size: {composite_image.shape}")
    
    split_points = find_split_points(
        composite_image, 
        min_height=min_height, 
        tolerance=tolerance, 
        carving_step=carving_step
    )
    
    logging.info(f"Found {len(split_points)} splitting points")
    
    segments = []
    segment_boundaries = []
    
    prev_y = 0
    for y in split_points:
        segment = composite_image[prev_y:y, :, :]
        segments.append(segment)
        segment_boundaries.append((prev_y, y))
        prev_y = y

    if prev_y < total_height:
        segment = composite_image[prev_y:total_height, :, :]
        segments.append(segment)
        segment_boundaries.append((prev_y, total_height))

    if save_crops:
        for i, segment in enumerate(segments):
            output_file = os.path.join(output_path, f"segment_{i:03d}.png")
            cv2.imwrite(output_file, segment)
            logging.info(f"Saved segment to: {output_file}")
    
    result = {
        'original_path': input_path,
        'segments': segments,
        'segment_boundaries': segment_boundaries,
        'composite_height': total_height,
        'composite_width': max_width,
    }
    
    return result


def find_split_points(
    image: np.ndarray, 
    min_height: int, 
    tolerance: float, 
    carving_step: int
) -> List[int]:
    """
    Find natural split points in the image using the seam carving algorithm.
    
    Args:
        image: Composite image to split
        min_height: Minimum height between split points
        tolerance: Threshold for color homogeneity
        carving_step: Window size for finding local maxima
        
    Returns:
        List of y-coordinates where the image should be split
    """
    height, width, _ = image.shape
    split_points = []

    y = min_height
    
    while y < height - min_height:
        candidate_y = y
        homogeneity = calculate_color_homogeneity(image, candidate_y, width)

        if homogeneity >= tolerance:
            split_points.append(candidate_y)
            y += min_height
        else:
            y += carving_step
        
    return split_points


def calculate_color_homogeneity(image: np.ndarray, y: int, width: int, theta: float = 0.1) -> float:
    """
    Share of horizontally adjacent pixel pairs in row y that differ by less than
    10 (Euclidean RGB distance), or 0 if any pair differs by 50 or more.

    Vectorised; returns the same values as the original per-pixel loop.

    Args:
        image: Input image
        y: Y-coordinate of the row to analyze
        width: Width of the image
        theta: Unused, kept for compatibility

    Returns:
        Homogeneity coefficient between 0 and 1
    """
    if width < 2:
        return 0.0
    row = image[y, :width, :].astype(float)
    diff = np.sqrt(np.sum(np.square(row[1:] - row[:-1]), axis=1))
    if diff.max() >= 50.0:
        return 0.0
    return float(np.count_nonzero(diff < 10.0) / (width - 1))