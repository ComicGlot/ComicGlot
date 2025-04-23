#!/usr/bin/env python3
"""
Bubble detection script for ComicGlot.

This script provides a command-line interface for detecting text bubbles 
in comic images. It can be used to process individual images or entire folders
of images, and supports various detection models and options.

Example usage:
    python -m comicglot.scripts.bubble_detection --input path/to/image.jpg --output path/to/output
    python -m comicglot.scripts.bubble_detection --input path/to/folder --output path/to/output --recursive
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.detection import YOLODetector, DETRDetector
from comicglot.utils.file_utils import load_image, save_image
from comicglot.utils.data_types import TextBlock
from comicglot.utils.spatial_processor import process_text_blocks
from comicglot.utils.visualization import draw_boxes_with_labels, visualize_separator_lines

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_single_image(
    image_path: str,
    output_path: Optional[str] = None,
    detection_model: str = 'yolo',
    model_path: Optional[str] = None,
    confidence: float = 0.5,
    iou: float = 0.7,
    save_visualization: bool = True,
    apply_spatial_processing: bool = True,
) -> Dict:
    """
    Process a single image to detect text bubbles.
    
    Args:
        image_path: Path to the input image
        output_path: Base path for output files
        detection_model: Detection model to use ('yolo' or 'detr')
        model_path: Path to custom model weights (optional)
        confidence: Confidence threshold for detections
        iou: IoU threshold for Non-Maximum Suppression
        save_visualization: Whether to save visualization images
        apply_spatial_processing: Whether to apply spatial processing to rank
                                  bubbles and identify separators
        
    Returns:
        Dictionary containing detection results and metadata
    """
    logger.info(f"Processing image: {image_path}")

    if detection_model.lower() == 'yolo':
        if model_path is None:
            model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
        detector = YOLODetector(model_path)
    elif detection_model.lower() == 'detr':
        if model_path is None:
            model_path = os.path.abspath(os.getenv("DETR_MODEL_PATH", "./models/DETR-RESNET-50"))
        detector = DETRDetector(model_path)
    else:
        raise ValueError(f"Unsupported detection model: {detection_model}")

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
        detection_results = detections
        sorted_blocks = text_blocks
        separator_lines = [None] * len(text_blocks)

    results = {
        "image_path": image_path,
        "processed_time": logging.Formatter.converter(),
        "detection_model": detection_model,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "spatial_processing_applied": apply_spatial_processing,
        "detections": detection_results
    }

    if output_path:
        os.makedirs(output_path, exist_ok=True)

        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]

        json_path = os.path.join(output_path, f"{base_name}_detections.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved results to: {json_path}")

        if save_visualization:
            viz_image = image.copy()

            boxes = [block.box for block in sorted_blocks]
            classes = [block.class_name for block in sorted_blocks]
            confidences = [block.confidence for block in sorted_blocks]
            ids = [block.id for block in sorted_blocks]
            
            viz_image = draw_boxes_with_labels(viz_image, boxes, classes, confidences, ids)
            
            if apply_spatial_processing:
                viz_image = visualize_separator_lines(viz_image, sorted_blocks, separator_lines)

            viz_path = os.path.join(output_path, f"{base_name}_detection_viz.jpg")
            save_image(viz_image, viz_path)
            logger.info(f"Saved visualization to: {viz_path}")
    
    return results

def process_folder(
    input_folder: str,
    output_folder: Optional[str] = None,
    recursive: bool = False,
    **kwargs
) -> Dict[str, Dict]:
    """
    Process all images in a folder to detect text bubbles.
    
    Args:
        input_folder: Path to the folder containing images
        output_folder: Path to the output folder
        recursive: Whether to search for images recursively
        **kwargs: Additional arguments to pass to process_single_image
        
    Returns:
        Dictionary mapping image paths to their detection results
    """
    logger.info(f"Processing folder: {input_folder}")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

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

    results = {}
    for image_path in sorted(image_files):
        rel_path = os.path.relpath(image_path, input_folder)

        if recursive and output_folder:
            rel_dir = os.path.dirname(rel_path)
            img_output_folder = os.path.join(output_folder, rel_dir)
            os.makedirs(img_output_folder, exist_ok=True)
        else:
            img_output_folder = output_folder
        
        try:
            image_results = process_single_image(
                image_path=image_path,
                output_path=img_output_folder,
                **kwargs
            )
            results[rel_path] = image_results
            logger.info(f"Processed: {rel_path}")
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {str(e)}")
            results[rel_path] = {"error": str(e)}

    if output_folder:
        summary_path = os.path.join(output_folder, "detection_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Create a simplified summary to avoid huge files
            summary = {
                k: {
                    "detections_count": len(v.get("detections", [])),
                    "error": v.get("error")
                }
                for k, v in results.items()
            }
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary to: {summary_path}")
    
    return results

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
    
    args = parser.parse_args()
    
    try:
        if os.path.isfile(args.input):
            process_single_image(
                image_path=args.input,
                output_path=args.output,
                detection_model=args.model,
                model_path=args.model_path,
                confidence=args.confidence,
                iou=args.iou,
                save_visualization=not args.no_visualization,
                apply_spatial_processing=not args.no_spatial
            )
            logger.info("Successfully processed image")
            
        elif os.path.isdir(args.input):
            process_folder(
                input_folder=args.input,
                output_folder=args.output,
                recursive=args.recursive,
                detection_model=args.model,
                model_path=args.model_path,
                confidence=args.confidence,
                iou=args.iou,
                save_visualization=not args.no_visualization,
                apply_spatial_processing=not args.no_spatial
            )
            logger.info("Successfully processed folder")
            
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return 1
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return 1
    
    return 0

def run_direct():
    """Run the script with hard-coded parameters."""
    YOLO_MODEL_PATH = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
    INPUT_PATH = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/frames/"))
    OUTPUT_PATH = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    
    if os.path.isfile(INPUT_PATH):
        process_single_image(
            image_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            detection_model="yolo",
            model_path=YOLO_MODEL_PATH,
            confidence=0.5,
            iou=0.7,
            save_visualization=True,
            apply_spatial_processing=True
        )
    else:
        process_folder(
            input_folder=INPUT_PATH,
            output_folder=OUTPUT_PATH,
            recursive=False,
            detection_model="yolo",
            model_path=YOLO_MODEL_PATH,
            confidence=0.5,
            iou=0.7,
            save_visualization=True,
            apply_spatial_processing=True
        )
    
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