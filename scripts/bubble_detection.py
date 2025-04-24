"""
Bubble detection script for comic processing.

This script provides a standalone interface for detecting text bubbles
in comic images, incorporating spatial processing and visualization.
It can process individual images or entire folders of images.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
import numpy as np
from dotenv import load_dotenv
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.detection import YOLODetector, DETRDetector
from comicglot.utils.data_types import TextBlock
from comicglot.utils.spatial_processor import process_text_blocks
from comicglot.utils.visualization import draw_boxes_with_labels, visualize_separator_lines
from comicglot.utils.file_utils import load_image, save_image, get_next_run_folder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bubble_detection")

load_dotenv()

def process_single_image(
    image_path: str,
    detector,
    output_path: Optional[str] = None,
    confidence: float = 0.5,
    iou: float = 0.7,
    save_visualization: bool = True,
    save_crops: bool = False,
    apply_spatial_processing: bool = True,
) -> Dict:
    """
    Process a single image to detect text bubbles.
    
    Args:
        image_path: Path to the input image
        detector: Initialized detector (YOLODetector or DETRDetector)
        output_path: Base path for output files
        confidence: Confidence threshold for detections
        iou: IoU threshold for Non-Maximum Suppression
        save_visualization: Whether to save visualization images
        save_crops: Whether to save cropped text blocks
        apply_spatial_processing: Whether to apply spatial processing to rank
                                  bubbles and identify separators
        
    Returns:
        Dictionary containing detection results and metadata
    """
    image = load_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    detections = detector.process_image(image_path, conf=confidence, iou=iou)

    text_blocks = [
        TextBlock(
            box=np.array(block["box"]),
            class_name=block["class_name"],
            confidence=block["confidence"],
            id=block["id"]
        )
        for block in detections
    ]

    if apply_spatial_processing:
        sorted_blocks, separator_lines = process_text_blocks(text_blocks)

        processed_blocks = []
        for i, (block, separators) in enumerate(zip(sorted_blocks, separator_lines)):
            block_dict = {
                "id": i,
                "box": block.box.tolist(),
                "class_name": block.class_name,
                "confidence": block.confidence,
                "separators": [sep.tolist() for sep in separators] if separators else None
            }
            processed_blocks.append(block_dict)
        
        detection_results = processed_blocks
    else:
        detection_results = [block.to_dict() for block in text_blocks]
        sorted_blocks = text_blocks
        separator_lines = [None] * len(text_blocks)

    results = {
        "image_path": image_path,
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detections_count": len(detection_results),
        "detections": detection_results
    }

    if output_path:
        os.makedirs(output_path, exist_ok=True)

        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]

        if save_visualization:
            vis_image = image.copy()
            vis_dir = os.path.join(output_path, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            boxes = [block.box for block in sorted_blocks]
            classes = [block.class_name for block in sorted_blocks]
            confidences = [block.confidence for block in sorted_blocks]
            ids = [block.id for block in sorted_blocks]

            labels = [
                f"{ids[i]}: {classes[i]} ({confidences[i]:.2f})"
                for i in range(len(boxes))
            ]

            vis_image = draw_boxes_with_labels(vis_image, boxes, labels, classes)

            if apply_spatial_processing:
                vis_image = visualize_separator_lines(vis_image, sorted_blocks, separator_lines)

            viz_path = os.path.join(vis_dir, f"{base_name}_detection.jpg")
            save_image(vis_image, viz_path)

        if save_crops:
            crops_dir = os.path.join(output_path, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            
            for i, block in enumerate(sorted_blocks):
                x1, y1, x2, y2 = map(int, block.box)
                crop = image[y1:y2, x1:x2].copy()

                if apply_spatial_processing and separator_lines[i]:
                    from comicglot.utils.spatial_processor import apply_separator_masks

                    local_separators = []
                    for sep in separator_lines[i]:
                        local_p1 = np.array([sep[0][0] - x1, sep[0][1] - y1], dtype=np.int32)
                        local_p2 = np.array([sep[1][0] - x1, sep[1][1] - y1], dtype=np.int32)
                        local_sep = np.array([local_p1, local_p2])
                        local_separators.append(local_sep)
                    
                    crop = apply_separator_masks(crop, local_separators)
                
                crop_path = os.path.join(crops_dir, f"{base_name}_{i:03d}_{block.class_name}.jpg")
                save_image(crop, crop_path)
    
    return results

def process_folder(
    input_folder: str,
    detector,
    output_folder: Optional[str] = None,
    recursive: bool = False,
    confidence: float = 0.5,
    iou: float = 0.7,
    save_visualization: bool = True,
    save_crops: bool = False,
    apply_spatial_processing: bool = True
) -> Dict:
    """
    Process all images in a folder to detect text bubbles.
    
    Args:
        input_folder: Path to the folder containing images
        detector: Initialized detector (YOLODetector or DETRDetector)
        output_folder: Path to the output folder
        recursive: Whether to search for images recursively
        confidence: Confidence threshold for detections
        iou: IoU threshold for Non-Maximum Suppression
        save_visualization: Whether to save visualization images
        save_crops: Whether to save cropped text blocks
        apply_spatial_processing: Whether to apply spatial processing
        
    Returns:
        Dictionary with aggregated results
    """
    logger.info(f"Processing folder: {input_folder}")

    if output_folder:
        run_folder = get_next_run_folder(output_folder, "detection")
        logger.info(f"Saving outputs to: {run_folder}")
    else:
        run_folder = None

    if save_crops and run_folder:
        crops_dir = os.path.join(run_folder, "crops")
        os.makedirs(crops_dir, exist_ok=True)

    if save_visualization and run_folder:
        viz_dir = os.path.join(run_folder, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    if recursive:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_folder, file))
    
    logger.info(f"Found {len(image_files)} image(s)")

    image_files = sorted(image_files)

    all_results = {
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(image_files),
        "total_bubbles": 0,
        "class_counts": {},
        "results": {}
    }

    for image_path in image_files:
        rel_path = os.path.relpath(image_path, input_folder)
        
        try:
            image_results = process_single_image(
                image_path=image_path,
                detector=detector,
                output_path=run_folder,
                confidence=confidence,
                iou=iou,
                save_visualization=save_visualization,
                save_crops=save_crops,
                apply_spatial_processing=apply_spatial_processing
            )

            all_results["total_bubbles"] += len(image_results["detections"])
            if "full_detections" not in all_results:
                all_results["full_detections"] = {}
            all_results["full_detections"][rel_path] = image_results["detections"]

            for detection in image_results["detections"]:
                class_name = detection["class_name"]
                all_results["class_counts"][class_name] = all_results["class_counts"].get(class_name, 0) + 1

            all_results["results"][rel_path] = {
                "detection_count": len(image_results["detections"]),
                "classes": [d["class_name"] for d in image_results["detections"]]
            }
            
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {str(e)}")
            all_results["results"][rel_path] = {"error": str(e)}

    if run_folder:
        summary_path = os.path.join(run_folder, "detection_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        boxes_txt_path = os.path.join(run_folder, "detected_boxes.txt")
        with open(boxes_txt_path, 'w', encoding='utf-8') as f:
            for image_path, image_result in all_results["results"].items():
                if "error" in image_result:
                    continue
                    
                f.write(f"Image: {image_path}\n")
                f.write(f"Total detections: {image_result['detection_count']}\n")

                detections = all_results.get("full_detections", {}).get(image_path, [])
                for detection in detections:
                    box = detection["box"]
                    f.write(f"Box {detection['id']}: id={detection['id']}, class={detection['class_name']}, ")
                    f.write(f"conf={detection['confidence']:.4f}, ")
                    f.write(f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")
                f.write("\n")

        logger.info(f"Detected {all_results['total_bubbles']} bubbles across {len(image_files)} images")


    return all_results

def run_with_args():
    """Run the script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect text bubbles in comic images")
    
    # Input/output arguments
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or folder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output folder")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Process folders recursively")
    
    # Detection configuration
    parser.add_argument("--model", "-m", type=str, default="yolo",
                        choices=["yolo", "detr"],
                        help="Detection model to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for Non-Maximum Suppression")
    
    # Processing options
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial processing (bubble ranking and separator detection)")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Don't save visualization images")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save cropped text blocks")
    
    args = parser.parse_args()
    
    try:
        if args.model.lower() == 'yolo':
            if args.model_path is None:
                model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
            else:
                model_path = args.model_path
            detector = YOLODetector(model_path)
        elif args.model.lower() == 'detr':
            if args.model_path is None:
                model_path = os.path.abspath(os.getenv("DETR_MODEL_PATH", "./models/DETR-RESNET-50"))
            else:
                model_path = args.model_path
            detector = DETRDetector(model_path)
        else:
            raise ValueError(f"Unsupported detection model: {args.model}")

        if os.path.isfile(args.input):
            if args.output:
                run_folder = get_next_run_folder(args.output, "detection")
                logger.info(f"Saving outputs to: {run_folder}")
            else:
                run_folder = None

            results = process_single_image(
                image_path=args.input,
                detector=detector,
                output_path=run_folder,
                confidence=args.confidence,
                iou=args.iou,
                save_visualization=not args.no_visualization,
                save_crops=args.save_crops,
                apply_spatial_processing=not args.no_spatial
            )

            if run_folder:
                results_path = os.path.join(run_folder, "detection_results.json")
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                boxes_txt_path = os.path.join(run_folder, "detected_boxes.txt")
                with open(boxes_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {args.input}\n")
                    f.write(f"Total detections: {len(results['detections'])}\n")
                    
                    for detection in results["detections"]:
                        box = detection["box"]
                        f.write(f"Box {detection['id']}: id={detection['id']}, class={detection['class_name']}, ")
                        f.write(f"conf={detection['confidence']:.4f}, ")
                        f.write(f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")
                        
                logger.info(f"Detected {len(results['detections'])} bubbles in the image")
                        
        elif os.path.isdir(args.input):
            process_folder(
                input_folder=args.input,
                detector=detector,
                output_folder=args.output,
                recursive=args.recursive,
                confidence=args.confidence,
                iou=args.iou,
                save_visualization=not args.no_visualization,
                save_crops=args.save_crops,
                apply_spatial_processing=not args.no_spatial
            )
            
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return 1
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_direct():
    """Run the script with hard-coded parameters from environment variables."""
    YOLO_MODEL_PATH = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
    INPUT_PATH = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/frames/"))
    OUTPUT_PATH = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    DETECTION_MODEL = os.getenv("DETECTION_MODEL", "yolo")
    CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU = float(os.getenv("IOU_THRESHOLD", "0.7"))
    
    logger.info(f"Starting processing with detection model: {DETECTION_MODEL}")
    logger.info(f"Input: {INPUT_PATH}, Output: {OUTPUT_PATH}")
    
    try:
        if DETECTION_MODEL.lower() == 'yolo':
            detector = YOLODetector(YOLO_MODEL_PATH)
        elif DETECTION_MODEL.lower() == 'detr':
            detector = DETRDetector(os.getenv("DETR_MODEL_PATH", "./models/DETR-RESNET-50"))
        else:
            logger.error(f"Unsupported detection model: {DETECTION_MODEL}")
            return 1
        
        if os.path.isfile(INPUT_PATH):
            run_folder = get_next_run_folder(OUTPUT_PATH, "detection")
            logger.info(f"Saving outputs to: {run_folder}")

            results = process_single_image(
                image_path=INPUT_PATH,
                detector=detector,
                output_path=run_folder,
                confidence=CONFIDENCE,
                iou=IOU,
                save_visualization=True,
                save_crops=True,
                apply_spatial_processing=True
            )

            results_path = os.path.join(run_folder, "detection_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            boxes_txt_path = os.path.join(run_folder, "detected_boxes.txt")
            with open(boxes_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {INPUT_PATH}\n")
                f.write(f"Total detections: {len(results['detections'])}\n")
                
                for detection in results['detections']:
                    box = detection["box"]
                    f.write(f"Box {detection['id']}: id={detection['id']}, class={detection['class_name']}, ")
                    f.write(f"conf={detection['confidence']:.4f}, ")
                    f.write(f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")
                
            logger.info(f"Detected {len(results['detections'])} bubbles in the image")     
                       
        elif os.path.isdir(INPUT_PATH):
            process_folder(
                input_folder=INPUT_PATH,
                detector=detector,
                output_folder=OUTPUT_PATH,
                recursive=False,
                confidence=CONFIDENCE,
                iou=IOU,
                save_visualization=True,
                save_crops=True,
                apply_spatial_processing=True
            )
            
        else:
            logger.error(f"Input path does not exist: {INPUT_PATH}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

def main():
    """Entry point function that determines how to run the script."""
    # Check if we're running with command-line arguments
    if len(sys.argv) > 1:
        return run_with_args()
    else:
        # No command line args, use hard-coded values
        return run_direct()

if __name__ == "__main__":
    sys.exit(main())