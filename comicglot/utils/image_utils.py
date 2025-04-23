import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

from comicglot.utils.file_utils import load_image

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                 preserve_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: Input image as numpy array
        width: Target width (if None, calculated from height and aspect ratio)
        height: Target height (if None, calculated from width and aspect ratio)
        preserve_aspect_ratio: Whether to preserve the original aspect ratio
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if preserve_aspect_ratio:
        if width is None:
            aspect = w / h
            width = int(height * aspect)
        elif height is None:
            aspect = h / w
            height = int(width * aspect)
        else:
            # Both dimensions specified but we need to preserve aspect ratio
            # We'll fit within the specified dimensions
            target_aspect = width / height
            current_aspect = w / h
            
            if current_aspect > target_aspect:
                # Image is wider than target, scale by width
                new_width = width
                new_height = int(width / current_aspect)
            else:
                # Image is taller than target, scale by height
                new_height = height
                new_width = int(height * current_aspect)
                
            width, height = new_width, new_height
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

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
    
    for file in sorted(os.listdir(input_path)):
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
            # Pad image with black to match max width
            padded_img = np.zeros((h, max_width, 3), dtype=np.uint8)
            padded_img[:, :w, :] = img
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
    Calculate color homogeneity
    
    Args:
        image: Input image
        y: Y-coordinate of the row to analyze
        width: Width of the image
        theta: Threshold parameter for color similarity (default: 0.1)
        
    Returns:
        Homogeneity coefficient between 0 and 1
    """

    row = image[y, :, :].astype(float)
    
    # Pixels with RGB differences less than 10 (out of 255) are considered similar
    absolute_threshold = 10.0
    
    similar_pixels_count = 0
    total_comparisons = width - 1
    
    max_diff = 0
    
    for x in range(width - 1):

        pixel_diff = np.sqrt(np.sum(np.square(row[x] - row[x+1])))
        max_diff = max(max_diff, pixel_diff)
        
        if pixel_diff < absolute_threshold:
            similar_pixels_count += 1
    
    base_homogeneity = similar_pixels_count / total_comparisons
    large_diff_penalty = 1.0 if max_diff < 50.0 else 0.0
    
    homogeneity = base_homogeneity * large_diff_penalty
    
    return homogeneity


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


def create_empty_image(height: int, width: int, channels: int = 3) -> np.ndarray:
    """
    Create an empty (black) image with specified dimensions.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        channels: Number of color channels (3 for RGB, 1 for grayscale)
        
    Returns:
        Empty image as numpy array
    """
    return np.zeros((height, width, channels), dtype=np.uint8)