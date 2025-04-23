#!/usr/bin/env python3
"""
Comic processing script for ComicGlot.

This script provides a command-line interface for the full ComicGlot pipeline,
including detection, recognition, and translation of text in comic images.

Example usage:
    python -m comicglot.scripts.process_comic --input path/to/image.jpg --output path/to/output
    python -m comicglot.scripts.process_comic --input path/to/chapter --output path/to/output --batch
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dotenv import load_dotenv

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.pipeline.comic_processor import ComicProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_single(
    input_path: str,
    output_path: Optional[str],
    detection_model: str,
    recognition_model: str,
    translator: Optional[str],
    source_lang: str,
    target_lang: str,
    detection_model_path: Optional[str],
    recognition_model_path: Optional[str],
    save_intermediates: bool
) -> Dict[str, Any]:
    """
    Process a single image or chapter.
    
    Args:
        input_path: Path to input image or chapter directory
        output_path: Path to output directory
        detection_model: Detection model to use
        recognition_model: Recognition model to use
        translator: Translator to use (None for no translation)
        source_lang: Source language code
        target_lang: Target language code
        detection_model_path: Path to custom detection model
        recognition_model_path: Path to custom recognition model
        save_intermediates: Whether to save intermediate results
        
    Returns:
        Processing results
    """
    # Initialize processor
    processor = ComicProcessor(
        detection_model=detection_model,
        recognition_model=recognition_model,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path
    )
    
    # Check if input is a file or directory
    if os.path.isfile(input_path):
        logger.info(f"Processing single image: {input_path}")
        results = processor.process_image(input_path, output_path, save_intermediates)
    else:
        logger.info(f"Processing chapter: {input_path}")
        results = processor.process_chapter(input_path, output_path, save_intermediates)
    
    return results

def process_batch(
    input_path: str,
    output_path: Optional[str],
    detection_model: str,
    recognition_model: str,
    translator: Optional[str],
    source_lang: str,
    target_lang: str,
    detection_model_path: Optional[str],
    recognition_model_path: Optional[str],
    save_intermediates: bool
) -> Dict[str, Any]:
    """
    Process multiple chapters in batch mode.
    
    Args:
        input_path: Path to base directory containing chapter folders
        output_path: Path to output base directory
        detection_model: Detection model to use
        recognition_model: Recognition model to use
        translator: Translator to use (None for no translation)
        source_lang: Source language code
        target_lang: Target language code
        detection_model_path: Path to custom detection model
        recognition_model_path: Path to custom recognition model
        save_intermediates: Whether to save intermediate results
        
    Returns:
        Batch processing results
    """
    # Initialize processor
    processor = ComicProcessor(
        detection_model=detection_model,
        recognition_model=recognition_model,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path
    )
    
    logger.info(f"Processing chapters in batch mode: {input_path}")
    results = processor.batch_process_chapters(input_path, output_path, save_intermediates)
    
    return results

def run_with_args():
    """Run the script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Process comic images with ComicGlot")
    
    # Input/output arguments
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image, chapter directory, or base directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output directory")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process in batch mode (input is a base directory containing chapters)")
    
    # Model configuration
    parser.add_argument("--detection", "-d", type=str, default="yolo",
                        choices=["yolo", "detr"],
                        help="Detection model to use")
    parser.add_argument("--recognition", "-r", type=str, default="trocr-korean",
                        choices=["trocr-korean", "trocr-base", "easyocr"],
                        help="Recognition model to use")
    parser.add_argument("--translator", "-t", type=str, default=None,
                        choices=[None, "claude"],
                        help="Translator to use (None for no translation)")
    
    # Language configuration
    parser.add_argument("--source-lang", "-s", type=str, default="ko",
                        help="Source language code")
    parser.add_argument("--target-lang", "-tl", type=str, default="en",
                        help="Target language code")
    
    # Advanced options
    parser.add_argument("--detection-model-path", type=str, default=None,
                        help="Path to custom detection model")
    parser.add_argument("--recognition-model-path", type=str, default=None,
                        help="Path to custom recognition model")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save intermediate results")
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            process_batch(
                input_path=args.input,
                output_path=args.output,
                detection_model=args.detection,
                recognition_model=args.recognition,
                translator=args.translator,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                detection_model_path=args.detection_model_path,
                recognition_model_path=args.recognition_model_path,
                save_intermediates=args.save_intermediates
            )
            logger.info("Successfully processed chapters in batch mode")
            
        else:
            process_single(
                input_path=args.input,
                output_path=args.output,
                detection_model=args.detection,
                recognition_model=args.recognition,
                translator=args.translator,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                detection_model_path=args.detection_model_path,
                recognition_model_path=args.recognition_model_path,
                save_intermediates=args.save_intermediates
            )
            logger.info("Successfully processed input")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return 1
    
    return 0

def run_direct():
    """Run the script with hard-coded parameters from environment variables."""
    # Load parameters from environment variables
    input_path = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/chapters/"))
    output_path = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    detection_model = os.getenv("DETECTION_MODEL", "yolo")
    recognition_model = os.getenv("RECOGNITION_MODEL", "trocr-korean")
    translator = os.getenv("TRANSLATOR", None)
    source_lang = os.getenv("SOURCE_LANG", "ko")
    target_lang = os.getenv("TARGET_LANG", "en")
    detection_model_path = os.getenv("YOLO_MODEL_PATH")
    recognition_model_path = os.getenv("TROCR_MODEL_PATH")
    
    # Use batch mode if input is a directory with chapter subdirectories
    is_batch_mode = False
    if os.path.isdir(input_path):
        # Check if there are subdirectories that could be chapters
        subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        if subdirs and all(any(os.path.isdir(os.path.join(input_path, d, subdir)) for subdir in subdirs) for d in ['images', 'text']):
            is_batch_mode = True
    
    logger.info(f"Starting processing with detection={detection_model}, recognition={recognition_model}")
    logger.info(f"Input: {input_path}, Output: {output_path}")
    
    try:
        if is_batch_mode:
            logger.info("Running in batch mode")
            process_batch(
                input_path=input_path,
                output_path=output_path,
                detection_model=detection_model,
                recognition_model=recognition_model,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                detection_model_path=detection_model_path,
                recognition_model_path=recognition_model_path,
                save_intermediates=True
            )
        else:
            logger.info("Running in single mode")
            process_single(
                input_path=input_path,
                output_path=output_path,
                detection_model=detection_model,
                recognition_model=recognition_model,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                detection_model_path=detection_model_path,
                recognition_model_path=recognition_model_path,
                save_intermediates=True
            )
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return 1

def main():
    # Check if we're running with command-line arguments
    if len(sys.argv) > 1:
        return run_with_args()
    else:
        # No command line args, use hard-coded values
        return run_direct()

if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------
# def run_with_args():
#     """Run the script with command-line arguments."""
#     parser = argparse.ArgumentParser(description="OCR Processing for Comics")
    
#     # Input/output arguments
#     parser.add_argument("--input", "-i", type=str, required=True,
#                         help="Path to input image or folder")
#     parser.add_argument("--output", "-o", type=str, default=None,
#                         help="Path to output folder")
#     parser.add_argument("--batch", "-b", action="store_true",
#                         help="Process in batch mode (input is a base directory containing chapters)")
    
#     # Model configuration
#     parser.add_argument("--detection-model", type=str, default="yolo", choices=["yolo", "detr"],
#                         help="Type of detection model to use (yolo or detr)")
#     parser.add_argument("--detection-model-path", type=str, default=None,
#                         help="Path to detection model (defaults to env var YOLO_MODEL_PATH or DETR_MODEL_PATH)")
#     parser.add_argument("--trocr-model", type=str, default=None,
#                         help="Path to TrOCR model (defaults to env var TROCR_MODEL_PATH)")
#     parser.add_argument("--ocr-languages", "-l", type=str, default="ko,en",
#                         help="Comma-separated list of language codes (e.g., 'ko,en')")
    
#     # Processing options
#     parser.add_argument("--segmentation", "-s", action="store_true",
#                         help="Use segmentation for long comics")
#     parser.add_argument("--min-seg-height", type=int, default=800,
#                         help="Minimum segment height for segmentation")
#     parser.add_argument("--save-segments", action="store_true",
#                         help="Save segmented images")
#     parser.add_argument("--min-block-size", type=int, default=20,
#                         help="Minimum size of text blocks to detect")
#     parser.add_argument("--confidence", "-c", type=float, default=0.5,
#                         help="Confidence threshold for detection")
#     parser.add_argument("--iou", type=float, default=0.7,
#                         help="IoU threshold for NMS")
    
#     # Output options
#     parser.add_argument("--no-visualization", action="store_true",
#                         help="Don't save visualization images")
#     parser.add_argument("--save-crops", action="store_true",
#                         help="Save cropped text blocks")
#     parser.add_argument("--save-lines", action="store_true",
#                         help="Save detected text line crops")
    
#     args = parser.parse_args()
    
#     # Load environment variable defaults if not specified
#     detection_model_path = args.detection_model_path
#     if not detection_model_path:
#         if args.detection_model.lower() == "yolo":
#             detection_model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
#         else:
#             detection_model_path = os.path.abspath(os.getenv("DETR_MODEL_PATH", "./models/DETR-RESNET-50"))
    
#     trocr_model_path = args.trocr_model
#     if not trocr_model_path:
#         trocr_model_path = os.path.abspath(os.getenv("TROCR_MODEL_PATH", None))
    
#     # Convert languages to list
#     ocr_languages = args.ocr_languages.split(',')
    
#     # Common arguments for processing
#     process_kwargs = {
#         "detection_model": args.detection_model,
#         "detection_model_path": detection_model_path,
#         "trocr_model_path": trocr_model_path,
#         "ocr_languages": ocr_languages,
#         "min_block_size": args.min_block_size,
#         "conf_threshold": args.confidence,
#         "iou_threshold": args.iou,
#         "save_visualization": not args.no_visualization,
#         "save_crops": args.save_crops,
#         "save_lines": args.save_lines,
#         "use_segmentation": args.segmentation,
#         "min_seg_height": args.min_seg_height,
#         "save_segments": args.save_segments
#     }
    
#     try:
#         # Check if input is a file or directory
#         if os.path.isfile(args.input):
#             logger.info(f"Processing single image: {args.input}")
#             result = process_image(
#                 args.input,
#                 args.output,
#                 **process_kwargs
#             )
            
#             if result["status"] == "success":
#                 logger.info(f"Successfully processed image with {len(result.get('blocks', []))} text blocks")
#                 return 0
#             else:
#                 logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
#                 return 1
                
#         elif os.path.isdir(args.input):
#             if args.batch:
#                 logger.info(f"Processing chapters in batch mode from: {args.input}")
#                 results = batch_process_chapters(
#                     args.input,
#                     args.output,
#                     **process_kwargs
#                 )
                
#                 # Count successful chapters
#                 success_count = sum(1 for c in results.values() 
#                                     if c.get("type") == "chapter" and c.get("status") == "success")
#                 total_chapters = sum(1 for c in results.values() if c.get("type") == "chapter")
                
#                 logger.info(f"Successfully processed {success_count}/{total_chapters} chapters")
                
#                 if success_count == 0 and total_chapters > 0:
#                     logger.error("No chapters were processed successfully")
#                     return 1
                    
#                 return 0
#             else:
#                 logger.info(f"Processing folder: {args.input}")
#                 results = segment_and_process_folder(
#                     args.input,
#                     args.output,
#                     **process_kwargs
#                 )
                
#                 # Count successful images
#                 success_count = sum(1 for r in results if r.get("status") == "success")
                
#                 logger.info(f"Successfully processed {success_count}/{len(results)} images")
                
#                 if success_count == 0 and len(results) > 0:
#                     logger.error("No images were processed successfully")
#                     return 1
                    
#                 return 0
#         else:
#             logger.error(f"Input path does not exist: {args.input}")
#             return 1
            
#     except Exception as e:
#         logger.error(f"Error during processing: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1

# def run_direct():
#     """Run the script with hard-coded parameters from environment variables."""
#     # Load basic parameters from environment variables
#     detection_model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
#     trocr_model_path = os.path.abspath(os.getenv("TROCR_MODEL_PATH", None))
#     input_path = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/chapters/test"))
#     output_path = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output"))
#     ocr_languages_str = os.getenv("LANGUAGE", "ko,en")
#     ocr_languages = ocr_languages_str.split(',')

#     # Determine if we're in batch mode
#     is_batch_mode = False
#     if os.path.isdir(input_path):
#         subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
#         if len(subdirs) > 0:
#             has_images = False
#             for subdir in subdirs:
#                 subdir_path = os.path.join(input_path, subdir)
#                 for file in os.listdir(subdir_path):
#                     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#                         has_images = True
#                         break
#                 if has_images:
#                     break
#             is_batch_mode = has_images
    
#     logger.info(f"Starting processing with detection model: {detection_model_path}")
#     logger.info(f"Input: {input_path}, Output: {output_path}")
#     logger.info(f"Batch mode: {is_batch_mode}")
    
#     # Common kwargs for processing
#     process_kwargs = {
#         "detection_model": "yolo",            # Type of detection model to use ("yolo" or "detr")
#         "detection_model_path": detection_model_path,  # Path to the detection model weights
#         "trocr_model_path": trocr_model_path,  # Path to custom TrOCR model for text recognition
#         "ocr_languages": ocr_languages,       # List of language codes for OCR (e.g., ['ko', 'en'])
#         "min_block_size": 20,                 # Minimum size of text blocks to be detected (pixels)
#         "conf_threshold": 0.5,                # Confidence threshold for detection (0.0-1.0)
#         "iou_threshold": 0.7,                 # IoU threshold for NMS in detection
#         "save_visualization": True,           # Whether to save annotated images with detection boxes
#         "save_crops": True,                   # Whether to save cropped text block images
#         "save_lines": False,                  # Whether to save individual text line crops
#         "use_segmentation": False,            # Whether to use segmentation for long comics
#         "min_seg_height": 800,                # Minimum height of segments (pixels) when segmentation is enabled
#         "save_segments": False                # Whether to save segmented images to disk
#     }
    
#     try:
#         if os.path.isfile(input_path):
#             # Process single image
#             logger.info(f"Processing single image: {input_path}")
#             result = process_image(
#                 input_path,
#                 output_path,
#                 **process_kwargs
#             )
            
#             if result["status"] == "success":
#                 logger.info(f"Successfully processed image with {len(result.get('blocks', []))} text blocks")
#                 return 0
#             else:
#                 logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
#                 return 1
                
#         elif os.path.isdir(input_path):
#             if is_batch_mode:
#                 logger.info(f"Processing chapters in batch mode from: {input_path}")
#                 results = batch_process_chapters(
#                     input_path,
#                     output_path,
#                     **process_kwargs
#                 )
                
#                 logger.info(f"Batch processing complete")
#                 return 0
#             else:
#                 logger.info(f"Processing folder: {input_path}")
#                 results = segment_and_process_folder(
#                     input_path,
#                     output_path,
#                     **process_kwargs
#                 )
                
#                 success_count = sum(1 for r in results if r.get("status") == "success")
#                 logger.info(f"Successfully processed {success_count}/{len(results)} images")
#                 return 0
#         else:
#             logger.error(f"Input path does not exist: {input_path}")
#             return 1
            
#     except Exception as e:
#         logger.error(f"Error during processing: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1

# def main():
#     """Entry point function that determines how to run the script."""
#     # Check if we're running with command-line arguments
#     if len(sys.argv) > 1:
#         return run_with_args()
#     else:
#         # No command line args, use hard-coded values
#         return run_direct()

# if __name__ == "__main__":
#     sys.exit(main())
