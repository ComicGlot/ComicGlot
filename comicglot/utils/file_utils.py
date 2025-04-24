import cv2
import numpy as np
import logging
import os
from typing import Optional, Tuple, List, Union
from urllib.request import urlopen
from io import BytesIO
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path or URL.
    
    Args:
        image_path: Path to the image file or URL
        
    Returns:
        Loaded image as NumPy array or None if loading fails
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            response = urlopen(image_path)
            image_data = response.read()
            
            # Convert to numpy array using PIL
            pil_image = PILImage.open(BytesIO(image_data))
            image = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
        else:
            # Load from local file
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
            return image
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {str(e)}")
        return None

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to a file.
    
    Args:
        image: Image as NumPy array
        output_path: Path where to save the image
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        result = cv2.imwrite(output_path, image)
        if not result:
            logger.error(f"Failed to save image to {output_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        return False

def get_next_run_folder(base_output_folder: str, tag: str,) -> str:
    """
    Find the smallest available "run{number}" folder in the output directory.
    
    Args:
        base_output_folder (str): The base output directory.
    
    Returns:
        str: Path to the new "run{number}" folder.
    """
    run_number = 1
    while True:
        run_folder = os.path.join(base_output_folder, f"{tag}_run{run_number}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        run_number += 1