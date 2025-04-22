import os
import time
import glob
import json
import torch
import numpy as np
import cv2
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from transformers import DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

load_dotenv()

DETR_MODEL_PATH = os.getenv("DETR_MODEL_PATH", "facebook/detr-resnet-50")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt")
DATASET_DIR = "./training/detection/datasets/Manhua_bubbles.v7i.coco/test"
IMAGES_DIR = DATASET_DIR
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "_annotations.coco.json")
OUTPUT_DIR = "./training/detection/benchmark"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5  # IoU threshold for determining if a detection matches a ground truth box
MAX_IMAGES = 1000  # Maximum number of images to process for benchmarking

# Class names will be loaded from COCO file
CLASS_NAMES = []
os.makedirs(OUTPUT_DIR, exist_ok=True)

class YOLODetector:
    """
    Detector class based on YOLO for text block detection in comics.
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
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.logger.info("YOLO model loaded successfully")
        except ImportError:
            self.logger.error("Could not import ultralytics. Please install with: pip install ultralytics")
            raise
    
    def detect(
        self, 
        image_path: str, 
        conf: float = 0.5, 
        iou: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Detect text blocks in an image.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            iou: Intersection over Union threshold for NMS
            
        Returns:
            List of dictionaries containing detection information
        """

        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        results = self.model(image_path, conf=conf, iou=iou)
        
        predictions = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            class_mapping = result.names  # Class names dictionary
            
            for i, box in enumerate(boxes):
                cls_id = int(class_ids[i])
                cls_name = class_mapping.get(cls_id, "unknown")
                predictions.append({
                    "id": i,
                    "box": box.tolist(),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(confidences[i])
                })
        
        return predictions

class DETRDetector:
    """
    Detector class based on DETR for text block detection in comics.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the DETR detector.
        
        Args:
            model_path: Path to the DETR model weights
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loading DETR model from {model_path}")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_path).to(self.device)

        self.name_to_yolo_id = {}
        for i, name in enumerate(CLASS_NAMES):
            self.name_to_yolo_id[name] = i

        self.detr_id_to_yolo_id = {}
        for detr_id, detr_name in self.model.config.id2label.items():
            if not detr_name.startswith("LABEL_"):
                if detr_name in self.name_to_yolo_id:
                    self.detr_id_to_yolo_id[int(detr_id)] = self.name_to_yolo_id[detr_name]
        
        self.logger.info(f"DETR model loaded successfully on {self.device}")
    
    def detect(
        self, 
        image_path: str, 
        conf: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect text blocks in an image using DETR.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections
            
        Returns:
            List of dictionaries containing detection information
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Error loading image: {image_path}")
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.shape[:2]], dtype=torch.float32).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]

        predictions = []
        
        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box_list = [round(x.item(), 2) for x in box]
            label_id = label.item()
            detr_class_name = self.model.config.id2label.get(label_id, f"unknown_{label_id}")

            yolo_class_id = self.detr_id_to_yolo_id.get(label_id, -1)
            
            predictions.append({
                "id": i,
                "box": box_list,
                "class_id": yolo_class_id,  # YOLO compatible class ID
                "detr_class_id": label_id,  # Original DETR class ID
                "class_name": detr_class_name,
                "confidence": score.item()
            })
        
        return predictions

def load_coco_dataset(
    annotations_file: str
) -> Tuple[Dict[int, Dict], Dict[str, List[Dict]]]:
    """
    Load COCO format dataset.
    
    Args:
        annotations_file: Path to the COCO annotations JSON file
        
    Returns:
        Tuple containing:
            - Dictionary of image info (keyed by image_id)
            - Dictionary of annotations (keyed by image file name)
    """
    global CLASS_NAMES
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    logger.info(f"Loading COCO annotations from {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat for cat in coco_data['categories']}
    CLASS_NAMES = [categories[i]['name'] for i in sorted(categories.keys()) if i < len(categories)]
    logger.info(f"Loaded {len(CLASS_NAMES)} class names: {CLASS_NAMES}")

    images = {img['id']: img for img in coco_data['images']}
    logger.info(f"Loaded info for {len(images)} images")
    
    annotations_by_image = defaultdict(list)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        
        if image_id in images:
            file_name = images[image_id]['file_name']

            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]

            class_id = ann['category_id']
            class_name = categories[class_id]['name'] if class_id in categories else f"unknown_{class_id}"

            annotation = {
                "id": ann['id'],
                "box": bbox,
                "class_id": class_id - 1,  # Adjust to 0-based indexing for compatibility
                "class_name": class_name,
                "segmentation": ann.get('segmentation', [])
            }
            
            annotations_by_image[file_name].append(annotation)
    
    logger.info(f"Loaded {len(coco_data['annotations'])} annotations for {len(annotations_by_image)} images")
    
    return images, annotations_by_image

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box in format [x_min, y_min, x_max, y_max]
        box2: Second box in same format
        
    Returns:
        IoU value between 0 and 1
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def evaluate_detections(
    detections: List[Dict[str, Any]], 
    ground_truths: List[Dict[str, Any]], 
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> Dict[str, Any]:
    """
    Evaluate detections against ground truth.
    
    Args:
        detections: List of detection dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching
        class_agnostic: If True, ignore class labels when matching
        
    Returns:
        Dictionary with TP, FP, FN counts
    """
    gt_matched = [False] * len(ground_truths)
    
    true_positives = 0
    class_true_positives = defaultdict(int)
    class_false_positives = defaultdict(int)
    
    for detection in detections:
        det_box = detection["box"]
        det_class = detection["class_id"]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue

            if not class_agnostic and det_class != gt["class_id"] and det_class != -1:
                continue
                
            gt_box = gt["box"]
            iou = calculate_iou(det_box, gt_box)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            true_positives += 1

            gt_class = ground_truths[best_gt_idx]["class_id"]
            class_true_positives[gt_class] += 1
        else:
            # False positive
            if det_class != -1:  # Skip unknown class (-1) in per-class metrics
                class_false_positives[det_class] += 1

    false_negatives = sum(not matched for matched in gt_matched)

    class_false_negatives = defaultdict(int)
    for gt_idx, gt in enumerate(ground_truths):
        if not gt_matched[gt_idx]:
            class_false_negatives[gt["class_id"]] += 1
    
    # Calculate per-class metrics
    class_metrics = {}
    all_classes = set(list(class_true_positives.keys()) + 
                     list(class_false_positives.keys()) + 
                     list(class_false_negatives.keys()))
    
    for class_id in all_classes:
        tp = class_true_positives[class_id]
        fp = class_false_positives[class_id]
        fn = class_false_negatives[class_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_id] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    total_detections = len(detections)
    total_ground_truths = len(ground_truths)
    
    precision = true_positives / total_detections if total_detections > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": total_detections - true_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_class": class_metrics
    }

def visualize_detections(
    image_path: str,
    detections: List[Dict[str, Any]], 
    ground_truths: List[Dict[str, Any]] = None,
    output_path: str = None
) -> None:
    """
    Visualize detections and ground truths on an image.
    
    Args:
        image_path: Path to the input image
        detections: List of detection dictionaries
        ground_truths: List of ground truth dictionaries (optional)
        output_path: Path to save the visualization (if None, don't save)
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return

    viz_image = image.copy()
    
    if ground_truths:
        for gt in ground_truths:
            box = [int(coord) for coord in gt["box"]]
            class_name = gt["class_name"]

            cv2.rectangle(viz_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            cv2.putText(viz_image, class_name, (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if "segmentation" in gt and gt["segmentation"]:
                for seg in gt["segmentation"]:
                    # Convert to points for polylines
                    if len(seg) >= 6:  # Need at least 3 points (6 coordinates)
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.polylines(viz_image, [points], True, (0, 255, 0), 1)

    for det in detections:
        box = [int(coord) for coord in det["box"]]
        class_name = det["class_name"]
        confidence = det["confidence"]

        cv2.rectangle(viz_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(viz_image, label, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, viz_image)
        logger.info(f"Saved visualization to {output_path}")
    
    return viz_image

def run_benchmark():
    """
    Run benchmark comparing YOLO and DETR detectors.
    """
    logger.info("Starting benchmark...")

    logger.info(f"Dataset directory exists: {os.path.exists(DATASET_DIR)}")
    logger.info(f"Annotations file exists: {os.path.exists(ANNOTATIONS_FILE)}")

    try:
        images_info, annotations_by_image = load_coco_dataset(ANNOTATIONS_FILE)
        logger.info(f"Loaded annotations for {len(annotations_by_image)} images")
    except Exception as e:
        logger.error(f"Error loading COCO dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    image_paths = []
    for filename in annotations_by_image.keys():
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)

    image_paths = image_paths[:MAX_IMAGES]
    
    num_images = len(image_paths)
    logger.info(f"Found {num_images} images with annotations to process")
    
    if num_images == 0:
        logger.error(f"No images with annotations found")
        return

    try:
        yolo_detector = YOLODetector(YOLO_MODEL_PATH)
        detr_detector = DETRDetector(DETR_MODEL_PATH)
    except Exception as e:
        logger.error(f"Error initializing detectors: {e}")
        import traceback
        traceback.print_exc()
        return

    yolo_metrics = {
        "time": [],
        "detections": [],
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "per_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    }
    
    detr_metrics = {
        "time": [],
        "detections": [],
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "per_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    }

    logger.info("Processing images and evaluating detections...")
    valid_images_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing"):
        img_name = os.path.basename(img_path)

        ground_truths = annotations_by_image.get(img_name, [])
        
        if not ground_truths:
            logger.warning(f"No annotations found for {img_name}, skipping")
            continue

        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Could not read image: {img_path}, skipping")
            continue

        valid_images_count += 1
        
        # YOLO detection
        start_time = time.time()
        try:
            yolo_dets = yolo_detector.detect(img_path, conf=CONFIDENCE_THRESHOLD)
            yolo_metrics["detections"].append(len(yolo_dets))
        except Exception as e:
            logger.error(f"Error with YOLO detector on {img_path}: {e}")
            continue
            
        yolo_metrics["time"].append(time.time() - start_time)
        
        # DETR detection
        start_time = time.time()
        try:
            detr_dets = detr_detector.detect(img_path, conf=CONFIDENCE_THRESHOLD)
            detr_metrics["detections"].append(len(detr_dets))
        except Exception as e:
            logger.error(f"Error with DETR detector on {img_path}: {e}")
            continue
            
        detr_metrics["time"].append(time.time() - start_time)
        
        # Evaluate YOLO
        yolo_eval = evaluate_detections(yolo_dets, ground_truths, IOU_THRESHOLD)
        yolo_metrics["true_positives"] += yolo_eval["true_positives"]
        yolo_metrics["false_positives"] += yolo_eval["false_positives"]
        yolo_metrics["false_negatives"] += yolo_eval["false_negatives"]
        
        # Update per-class metrics for YOLO
        for class_id, metrics in yolo_eval["per_class"].items():
            yolo_metrics["per_class"][class_id]["tp"] += metrics["true_positives"]
            yolo_metrics["per_class"][class_id]["fp"] += metrics["false_positives"]
            yolo_metrics["per_class"][class_id]["fn"] += metrics["false_negatives"]
        
        # Evaluate DETR
        detr_eval = evaluate_detections(detr_dets, ground_truths, IOU_THRESHOLD)
        detr_metrics["true_positives"] += detr_eval["true_positives"]
        detr_metrics["false_positives"] += detr_eval["false_positives"]
        detr_metrics["false_negatives"] += detr_eval["false_negatives"]
        
        # Update per-class metrics for DETR
        for class_id, metrics in detr_eval["per_class"].items():
            detr_metrics["per_class"][class_id]["tp"] += metrics["true_positives"]
            detr_metrics["per_class"][class_id]["fp"] += metrics["false_positives"]
            detr_metrics["per_class"][class_id]["fn"] += metrics["false_negatives"]
        
        # Save visualization for first few images
        if valid_images_count <= 5:
            viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            yolo_viz_path = os.path.join(viz_dir, f"{os.path.splitext(img_name)[0]}_yolo.jpg")
            visualize_detections(img_path, yolo_dets, ground_truths, yolo_viz_path)

            detr_viz_path = os.path.join(viz_dir, f"{os.path.splitext(img_name)[0]}_detr.jpg")
            visualize_detections(img_path, detr_dets, ground_truths, detr_viz_path)

    logger.info(f"Successfully processed {valid_images_count} valid images")
    
    if valid_images_count == 0:
        logger.error("No valid images were successfully processed. Cannot calculate metrics.")
        return
    
    # YOLO metrics
    yolo_total_time = sum(yolo_metrics["time"])
    yolo_avg_time = yolo_total_time / valid_images_count
    yolo_avg_detections = np.mean(yolo_metrics["detections"]) if yolo_metrics["detections"] else 0
    
    yolo_tp = yolo_metrics["true_positives"]
    yolo_fp = yolo_metrics["false_positives"]
    yolo_fn = yolo_metrics["false_negatives"]
    
    yolo_precision = yolo_tp / (yolo_tp + yolo_fp) if (yolo_tp + yolo_fp) > 0 else 0
    yolo_recall = yolo_tp / (yolo_tp + yolo_fn) if (yolo_tp + yolo_fn) > 0 else 0
    yolo_f1 = 2 * (yolo_precision * yolo_recall) / (yolo_precision + yolo_recall) if (yolo_precision + yolo_recall) > 0 else 0
    
    # DETR metrics
    detr_total_time = sum(detr_metrics["time"])
    detr_avg_time = detr_total_time / valid_images_count
    detr_avg_detections = np.mean(detr_metrics["detections"]) if detr_metrics["detections"] else 0
    
    detr_tp = detr_metrics["true_positives"]
    detr_fp = detr_metrics["false_positives"]
    detr_fn = detr_metrics["false_negatives"]
    
    detr_precision = detr_tp / (detr_tp + detr_fp) if (detr_tp + detr_fp) > 0 else 0
    detr_recall = detr_tp / (detr_tp + detr_fn) if (detr_tp + detr_fn) > 0 else 0
    detr_f1 = 2 * (detr_precision * detr_recall) / (detr_precision + detr_recall) if (detr_precision + detr_recall) > 0 else 0
    
    # Log overall results
    logger.info("\n--- BENCHMARK RESULTS ---")
    logger.info(f"Processed {valid_images_count} valid images out of {num_images} total images")
    
    logger.info("\nYOLO Performance:")
    logger.info(f"  Speed: {1/yolo_avg_time:.2f} images/second ({yolo_avg_time:.4f} seconds/image)")
    logger.info(f"  Average detections: {yolo_avg_detections:.2f} per image")
    logger.info(f"  Precision: {yolo_precision:.4f}")
    logger.info(f"  Recall: {yolo_recall:.4f}")
    logger.info(f"  F1 Score: {yolo_f1:.4f}")
    logger.info(f"  True Positives: {yolo_tp}")
    logger.info(f"  False Positives: {yolo_fp}")
    logger.info(f"  False Negatives: {yolo_fn}")

    logger.info("\nDETR Performance:")
    logger.info(f"  Speed: {1/detr_avg_time:.2f} images/second ({detr_avg_time:.4f} seconds/image)")
    logger.info(f"  Average detections: {detr_avg_detections:.2f} per image")
    logger.info(f"  Precision: {detr_precision:.4f}")
    logger.info(f"  Recall: {detr_recall:.4f}")
    logger.info(f"  F1 Score: {detr_f1:.4f}")
    logger.info(f"  True Positives: {detr_tp}")
    logger.info(f"  False Positives: {detr_fp}")
    logger.info(f"  False Negatives: {detr_fn}")

    logger.info("\n--- PER-CLASS METRICS ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_file = os.path.join(OUTPUT_DIR, "precision_recall_results.txt")
    with open(result_file, "w") as f:
        f.write("--- BENCHMARK RESULTS ---\n")
        f.write(f"Processed {valid_images_count} valid images out of {num_images} total images\n\n")
        
        f.write("YOLO Performance:\n")
        f.write(f"  Speed: {1/yolo_avg_time:.2f} images/second ({yolo_avg_time:.4f} seconds/image)\n")
        f.write(f"  Average detections: {yolo_avg_detections:.2f} per image\n")
        f.write(f"  Precision: {yolo_precision:.4f}\n")
        f.write(f"  Recall: {yolo_recall:.4f}\n")
        f.write(f"  F1 Score: {yolo_f1:.4f}\n")
        f.write(f"  True Positives: {yolo_tp}\n")
        f.write(f"  False Positives: {yolo_fp}\n")
        f.write(f"  False Negatives: {yolo_fn}\n\n")
        
        f.write("DETR Performance:\n")
        f.write(f"  Speed: {1/detr_avg_time:.2f} images/second ({detr_avg_time:.4f} seconds/image)\n")
        f.write(f"  Average detections: {detr_avg_detections:.2f} per image\n")
        f.write(f"  Precision: {detr_precision:.4f}\n")
        f.write(f"  Recall: {detr_recall:.4f}\n")
        f.write(f"  F1 Score: {detr_f1:.4f}\n")
        f.write(f"  True Positives: {detr_tp}\n")
        f.write(f"  False Positives: {detr_fp}\n")
        f.write(f"  False Negatives: {detr_fn}\n\n")
        
        f.write("--- PER-CLASS METRICS ---\n")
        f.write("\nYOLO Per-Class Metrics:\n")
        f.write(f"{'Class ID':<8} {'Class Name':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'TP':<6} {'FP':<6} {'FN':<6}\n")
        f.write("-" * 80 + "\n")
        
        for class_id in sorted(yolo_metrics["per_class"].keys()):
            metrics = yolo_metrics["per_class"][class_id]
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"
            
            f.write(f"{class_id:<8} {class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {tp:<6} {fp:<6} {fn:<6}\n")
        
        f.write("\nDETR Per-Class Metrics:\n")
        f.write(f"{'Class ID':<8} {'Class Name':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'TP':<6} {'FP':<6} {'FN':<6}\n")
        f.write("-" * 80 + "\n")
        
        for class_id in sorted(detr_metrics["per_class"].keys()):
            metrics = detr_metrics["per_class"][class_id]
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"
            
            f.write(f"{class_id:<8} {class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {tp:<6} {fp:<6} {fn:<6}\n")

        f.write("\n--- MODEL COMPARISON ---\n")
        f.write(f"Speed: YOLO is {detr_avg_time/yolo_avg_time:.2f}x faster than DETR\n")
        f.write(f"Precision: {'YOLO' if yolo_precision > detr_precision else 'DETR'} is better by {abs(yolo_precision - detr_precision):.4f}\n")
        f.write(f"Recall: {'YOLO' if yolo_recall > detr_recall else 'DETR'} is better by {abs(yolo_recall - detr_recall):.4f}\n")
        f.write(f"F1 Score: {'YOLO' if yolo_f1 > detr_f1 else 'DETR'} is better by {abs(yolo_f1 - detr_f1):.4f}\n")
        
    logger.info(f"Results saved to {result_file}")

    csv_file = os.path.join(OUTPUT_DIR, "precision_recall_results.csv")
    with open(csv_file, "w") as f:
        f.write("Model,Class ID,Class Name,Precision,Recall,F1 Score,TP,FP,FN\n")

        for class_id in sorted(yolo_metrics["per_class"].keys()):
            metrics = yolo_metrics["per_class"][class_id]
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"
            
            f.write(f"YOLO,{class_id},\"{class_name}\",{precision:.4f},{recall:.4f},{f1:.4f},{tp},{fp},{fn}\n")

        f.write(f"YOLO,all,\"Overall\",{yolo_precision:.4f},{yolo_recall:.4f},{yolo_f1:.4f},{yolo_tp},{yolo_fp},{yolo_fn}\n")

        for class_id in sorted(detr_metrics["per_class"].keys()):
            metrics = detr_metrics["per_class"][class_id]
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"
            
            f.write(f"DETR,{class_id},\"{class_name}\",{precision:.4f},{recall:.4f},{f1:.4f},{tp},{fp},{fn}\n")

        f.write(f"DETR,all,\"Overall\",{detr_precision:.4f},{detr_recall:.4f},{detr_f1:.4f},{detr_tp},{detr_fp},{detr_fn}\n")
    
    logger.info(f"CSV results saved to {csv_file}")

if __name__ == "__main__":
    run_benchmark()