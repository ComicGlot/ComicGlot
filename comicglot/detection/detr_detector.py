"""
DETR detector module for ComicGlot.

This module provides functionality to detect text blocks in images
using the DETR (DEtection TRansformer) model from Facebook/Hugging Face.
"""

import os
import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
except ImportError:
    logging.error("Could not import transformers. Please install with: pip install transformers")
    raise

class DETRDetector:
    """
    Detector class based on DETR for text block detection in comics.
    
    This class wraps the DETR model and provides methods to detect
    text blocks in images, returning bounding boxes, class names,
    and confidence scores.
    """
    
    def __init__(self, model_path: str = "facebook/detr-resnet-50"):
        """
        Initialize the DETR detector.
        
        Args:
            model_path: Path to the DETR model weights folder or model name
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if os.path.exists(model_path) and not model_path.startswith(("http://", "https://")):
            self.logger.info(f"Loading DETR model from local path: {model_path}")
        else:
            self.logger.info(f"Loading DETR model: {model_path}")
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        try:
            self.processor = DetrImageProcessor.from_pretrained(model_path)
            self.model = DetrForObjectDetection.from_pretrained(model_path).to(self.device)
            self.logger.info("DETR model loaded successfully")
            
            # Get class names mapping
            self.id2label = self.model.config.id2label
            self.logger.info(f"Model has {len(self.id2label)} classes")
            
        except Exception as e:
            self.logger.error(f"Failed to load DETR model: {str(e)}")
            raise
    
    def detect(
        self, 
        image_path: str, 
        threshold: float = 0.9
    ) -> List[Tuple[np.ndarray, str, float]]:
        """
        Detect text blocks in an image.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections
            
        Returns:
            List of tuples, each containing (box, class_name, confidence)
        """
        import cv2  # Import here to avoid dependency for basic usage
        
        self.logger.info(f"Running DETR inference on {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and prepare image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image for DETR
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # Format results
        predictions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_np = np.array([
                box[0].item(),  # x_min
                box[1].item(),  # y_min
                box[2].item(),  # x_max
                box[3].item(),  # y_max
            ])
            class_name = self.id2label[label.item()]
            confidence = score.item()
            predictions.append((box_np, class_name, confidence))
        
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
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            
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