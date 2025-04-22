"""
YOLO detector module for ComicGlot.

This module provides functionality to detect text blocks in images
using the YOLO object detection model.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Could not import ultralytics. Please install with: pip install ultralytics")
    raise

class YOLODetector:
    """
    Detector class based on YOLO for text block detection in comics.
    
    This class wraps the YOLO model and provides methods to detect
    text blocks in images, returning bounding boxes, class names,
    and confidence scores.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model weights file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.logger.info("YOLO model loaded successfully")
    
    def detect(
        self, 
        image_path: str, 
        conf: float = 0.5, 
        iou: float = 0.7
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """
        Detect text blocks in an image.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            iou: Intersection over Union threshold for NMS
            
        Returns:
            Tuple containing:
            - List of bounding boxes, each as [x1, y1, x2, y2]
            - List of class names
            - List of confidence scores
        """
        self.logger.info(f"Running YOLO inference on {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run YOLO inference
        results = self.model(image_path, conf=conf, iou=iou)
        
        # Extract detection results
        boxes = []
        class_names = []
        confidences = []
        
        for result in results:
            result_boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            result_confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            result_class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            class_mapping = result.names  # Class names dictionary
            
            for i, box in enumerate(result_boxes):
                cls_id = int(result_class_ids[i])
                cls_name = class_mapping.get(cls_id, "unknown")
                confidence = result_confidences[i]
                
                boxes.append(box)
                class_names.append(cls_name)
                confidences.append(confidence)
        
        self.logger.info(f"Detected {len(boxes)} text blocks")
        
        return boxes, class_names, confidences

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from dotenv import load_dotenv
    load_dotenv()
    
    YOLO_MODEL_PATH = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
    INPUT_IMAGE = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))
    
    detector = YOLODetector(YOLO_MODEL_PATH)
    boxes, classes, confidences = detector.detect(INPUT_IMAGE)
    
    for i, (box, cls_name, conf) in enumerate(zip(boxes, classes, confidences)):
        print(f"Detection {i+1}: {cls_name} ({conf:.4f}) at {box}")