import os
from dotenv import load_dotenv

import logging
from typing import List, Tuple
from ultralytics import YOLO
import numpy as np

class YOLOPredictor:
    def __init__(self, model_path: str):
        """
        Initializes the YOLO model for predictions.
        
        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.7) -> List[Tuple[np.ndarray, str, float]]:
        """
        Run YOLO predictions on an image.

        Args:
            image_path (str): Path to the image file.
            conf (float): Confidence threshold for predictions.
            iou (float): Intersection over Union threshold for predictions.

        Returns:
            List[Tuple[np.ndarray, str, float]]: A list of tuples, each containing
            - Bounding box coordinates (x1, y1, x2, y2)
            - Class name
            - Confidence score
        """
        logging.info(f"Running YOLO inference on {image_path}")

        # Run YOLO inference 
        #results = self.model(image_path, conf=conf, iou=iou) # TRYIN WITHOUT PARAMS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        results = self.model(image_path)
        predictions = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            class_names = result.names  # Class names dictionary

            for i, box in enumerate(boxes):
                cls_id = int(class_ids[i])
                cls_name = class_names.get(cls_id, "unknown")
                predictions.append((box, cls_name, confidences[i]))

        # logging.info("Predictions before ranking:")
        # for box, cls_name, conf in predictions:
        #     logging.info(f"Box: {box}, Class: {cls_name}, Confidence: {conf:.4f}")

        return self.rank_boxes(predictions)

    @staticmethod
    def rank_boxes(predictions: List[Tuple[np.ndarray, str, float]], y_tolerance_ratio: float = 0.5) -> List[Tuple[np.ndarray, str, float]]:
        """
        Rank bounding boxes based on Y-coordinates and rightmost X-coordinate for overlapping boxes.

        Args:
            predictions (List[Tuple[np.ndarray, str, float]]): List of predictions with boxes, class names, and confidence scores.
            y_tolerance_ratio (float): Tolerance as a ratio of the box height to consider boxes as overlapping.

        Returns:
            List[Tuple[np.ndarray, str, float]]: Ranked list of predictions.
            List[np.ndarray]: List of separator masks ('None' is passed if no separator needed).
        """
        if not predictions:
            return [], []
        print(predictions)
        # Sort the boxes first by Y-coordinate (`y1`) as a primary sort
        predictions.sort(key=lambda p: p[0][1])

        ranked = []
        separator_lines = [None] * len(predictions)  # Pre-fill with None

        while predictions:
            current = predictions.pop(0)
            current_box = current[0]
            x1, y1, x2, y2 = current_box
            #height = y2 - y1
            #y_tolerance = max(height * y_tolerance_ratio, 15)

            # Group boxes that are within the Y-tolerance range
            overlapping_group = [current]
            remaining = []
            group_y1 = y1
            group_y2 = y2

            for other in predictions:
                other_box = other[0]
                other_x1, other_y1, other_x2, other_y2 = other_box

                # Check if boxes overlap in Y-coordinate range
                if group_y1 <= other_y1 <= group_y2:
                    overlapping_group.append(other)
                    group_y2 = other_y2
                else:
                    remaining.append(other)

            # Update the remaining predictions
            predictions = remaining

            # Sort overlapping boxes by their X-coordinate (x1), ascending
            overlapping_group.sort(key=lambda p: p[0][0])
            
            # Check for X-axis overlaps and create separator lines
            for i in range(len(overlapping_group)):
                box1 = overlapping_group[i][0]
                x1, y1, x2, y2 = box1
                
                box_separator_lines = []
                
                for j in range(len(overlapping_group)):
                    box2 = overlapping_group[j][0]
                    other_x1, other_y1, other_x2, other_y2 = box2
                    if box1 == box2:
                        continue
                    
                    # Check for X-axis overlap
                    if x1 <= other_x1 <= x2:
                        # Calculate the separator line points
                        # Point 1: (max of left edges, min of top edges)
                        # Point 2: (min of right edges, max of bottom edges)
                        p1 = [max(x1, other_x1), min(x1, other_y1)]
                        p2 = [min(x2, other_x2), max(x2, other_y2)]
                        
                        
                        # Create separator line
                        separator_line = np.array([p1, p2], dtype=np.int32)
                        break
                
                ranked.append(overlapping_group[i])
                separator_lines.append(separator_line)

        # Make sure we have the same number of separator lines as ranked boxes
        while len(separator_lines) < len(ranked):
            separator_lines.append(None)

        return ranked, separator_lines


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        #level=logging.DEBUG,
    )

    # Load environment variables from .env file
    load_dotenv()

    YOLO_MODEL_PATH = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
    INPUT_IMAGE = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))

    predictor = YOLOPredictor(YOLO_MODEL_PATH)
    results = predictor.predict(INPUT_IMAGE)

    #logging.info("Predictions after ranking:")
    for box, cls_name, conf in results:
        logging.info(f"Box: {box}, Class: {cls_name}, Confidence: {conf:.4f}")
