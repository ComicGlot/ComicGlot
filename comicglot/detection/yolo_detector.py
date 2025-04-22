"""
YOLO detector module for ComicGlot.

This module provides functionality to detect text blocks in images
using the YOLO object detection model.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Union

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
    ) -> List[Tuple[np.ndarray, str, float]]:
        """
        Detect text blocks in an image.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            iou: Intersection over Union threshold for NMS
            
        Returns:
            List of tuples containing (box, class_name, confidence)
        """
        self.logger.info(f"Running YOLO inference on {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run YOLO inference
        results = self.model(image_path, conf=conf, iou=iou)
        
        # Extract detection results
        predictions = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            class_mapping = result.names  # Class names dictionary
            
            for i, box in enumerate(boxes):
                cls_id = int(class_ids[i])
                cls_name = class_mapping.get(cls_id, "unknown")
                predictions.append((box, cls_name, confidences[i]))
        
        self.logger.info(f"Detected {len(predictions)} text blocks")
        return predictions
    
    def process_image(
        self,
        image_path: str,
        conf: float = 0.5,
        iou: float = 0.7,
        return_visualization: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process an image to detect text blocks.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            iou: Intersection over Union threshold for NMS
            return_visualization: Unused parameter (kept for API compatibility)
            
        Returns:
            List of detected blocks as dictionaries
        """
        # Detect text blocks
        predictions = self.detect(image_path, conf, iou)
        
        # Convert to structured output
        results = []
        for i, (box, class_name, confidence) in enumerate(predictions):
            results.append({
                "id": i,
                "box": box.tolist(),
                "class_name": class_name,
                "confidence": float(confidence)
            })
        
        return results


if __name__ == "__main__":
    # Simple test code for the detector
    import cv2
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get paths from environment if available
    model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
    input_image = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))
    
    # Create detector and run detection
    detector = YOLODetector(model_path)
    results = detector.process_image(input_image)
    
    # Display results
    print(f"Detected {len(results)} text blocks:")
    for r in results:
        print(f"  ID: {r['id']}, Class: {r['class_name']}, Box: {r['box']}, Confidence: {r['confidence']:.4f}")