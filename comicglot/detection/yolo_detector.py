"""
YOLO detector module for ComicGlot.

This module provides functionality to detect text blocks in images
using the YOLO object detection model.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

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
    
    def visualize(
        self,
        image_path: str,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Visualize detected text blocks on an image.
        
        Args:
            image_path: Path to the image file
            detections: List of detected blocks as dictionaries
            
        Returns:
            Visualization image with bounding boxes
        """
        try:
            from comicglot.utils.visualization import draw_boxes_with_labels
            from comicglot.utils.file_utils import load_image
            
            # Load image
            image = load_image(image_path)
            
            # Draw bounding boxes
            boxes = [np.array(r["box"]) for r in detections]
            classes = [r["class_name"] for r in detections]
            confidences = [r["confidence"] for r in detections]
            ids = [r["id"] for r in detections]
            
            visualization = draw_boxes_with_labels(image, boxes, classes, confidences, ids)
            return visualization
            
        except ImportError:
            self.logger.warning("Visualization utilities not available, returning original image")
            import cv2
            return cv2.imread(image_path)

    def process_image(
        self,
        image_path: str,
        conf: float = 0.5,
        iou: float = 0.7,
        return_visualization: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Process an image to detect and rank text blocks, with optional visualization.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            iou: Intersection over Union threshold for NMS
            return_visualization: Whether to return a visualization image
            
        Returns:
            Tuple containing:
            - List of detected blocks as dictionaries
            - Visualization image (if return_visualization=True) or None
        """
        import cv2  # Import here to avoid dependency for basic usage
        
        # Detect text blocks
        ranked_predictions, separator_lines = self.detect(image_path, conf, iou)
        
        # Convert to structured output
        results = []
        for i, ((box, class_name, confidence), separator) in enumerate(zip(ranked_predictions, separator_lines)):
            results.append({
                "id": i,
                "box": box.tolist(),
                "class_name": class_name,
                "confidence": float(confidence),
                "separator": separator.tolist() if separator is not None else None
            })
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            try:
                from comicglot.utils.visualization import draw_boxes_with_labels, visualize_separator_lines
                from comicglot.utils.file_utils import load_image
                from comicglot.utils.data_types import TextBlock
                
                # Load image
                image = load_image(image_path)
                
                # Draw bounding boxes
                boxes = [np.array(r["box"]) for r in results]
                classes = [r["class_name"] for r in results]
                confidences = [r["confidence"] for r in results]
                ids = [r["id"] for r in results]
                
                visualization = draw_boxes_with_labels(image, boxes, classes, confidences, ids)
                
                # Draw separator lines
                text_blocks = [
                    TextBlock(box=np.array(r["box"]), class_name=r["class_name"], 
                              confidence=r["confidence"], id=r["id"])
                    for r in results
                ]
                sep_lines_list = []
                for r in results:
                    if r["separator"] is not None:
                        sep_lines_list.append([np.array(r["separator"])])
                    else:
                        sep_lines_list.append([])
                
                visualization = visualize_separator_lines(visualization, text_blocks, sep_lines_list)
                
            except ImportError:
                self.logger.warning("Visualization utilities not available, skipping visualization")
        
        return results, visualization