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
    if isinstance(image, str):
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