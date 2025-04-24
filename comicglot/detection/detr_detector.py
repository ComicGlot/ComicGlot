"""
DETR detector module for ComicGlot.

This module provides functionality to detect text blocks in images
using the DETR (DEtection TRansformer) model from Facebook/Hugging Face.
"""

import os
import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Union
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        try:
            # Always load the processor from the base model
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.logger.info("DETR processor loaded successfully")

            if os.path.exists(model_path) and not model_path.startswith(("http://", "https://")):
                self.logger.info(f"Loading DETR model from local path: {model_path}")
            else:
                self.logger.info(f"Loading DETR model: {model_path}")
                
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

        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]], dtype=torch.float32).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

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
    
    def process_image(
        self,
        image_path: str,
        threshold: float = 0.5,
        return_visualization: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process an image to detect text blocks.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections
            return_visualization: Unused parameter (kept for API compatibility)
            
        Returns:
            List of detected blocks as dictionaries
        """
        predictions = self.detect(image_path, threshold)

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
    import cv2
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    model_path = os.getenv("DETR_MODEL_PATH", "facebook/detr-resnet-50")
    input_image = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))

    detector = DETRDetector(model_path)
    results = detector.process_image(input_image)

    print(f"Detected {len(results)} text blocks:")
    for r in results:
        print(f"  ID: {r['id']}, Class: {r['class_name']}, Box: {r['box']}, Confidence: {r['confidence']:.4f}")