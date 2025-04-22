#!/usr/bin/env python3
"""
OCR processing script for ComicGlot.

This script provides a command-line interface for OCR and text recognition
in comic images. It can be used to process individual images or entire folders
of images, with various detection and recognition options.

Example usage:
    python -m comicglot.scripts.process_ocr --input path/to/image.jpg --output path/to/output
    python -m comicglot.scripts.process_ocr --input path/to/folder --output path/to/output --recursive
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.recognition import OCRProcessor
from comicglot.detection import YOLODetector
from comicglot.utils.file_utils import load_image, save_image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_single_image(
    image_path: str,
    output_path: Optional[str] = None,
    detection_model: str = 'craft',
    recognition_model: str = 'trocr',
    language: List[str] = ['ko', 'en'],
    custom_model_path: Optional[str] = None,
    save_visualization: bool = True,
    min_block_size: int = 20,
    paragraph_mode: bool = False
) -> Dict:
    """
    Process a single image with OCR.
    
    Args:
        image_path: Path to the input image
        output_path: Base path for output files
        detection_model: Text detection model ('craft' or 'east')
        recognition_model: Text recognition model ('trocr' or 'easyocr')
        language: List of language codes for OCR
        custom_model_path: Path to custom model weights (optional)
        save_visualization: Whether to save visualization images
        min_block_size: Minimum size of text blocks to be considered
        paragraph_mode: Whether to group lines into paragraphs
        
    Returns:
        Dictionary containing OCR results and metadata
    """
    logger.info(f"Processing image: {image_path}")
    
    # Initialize OCR processor
    ocr = OCRProcessor(
        language=language,
        detection_model=detection_model,
        recognition_model=recognition_model,
        custom_model_path=custom_model_path
    )
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Perform OCR
    ocr_results = ocr.perform_detection_and_recognition(
        image, 
        min_block_size=min_block_size,
        paragraph_mode=paragraph_mode
    )
    
    # Prepare results
    results = {
        "image_path": image_path,
        "processed_time": logging.Formatter.converter(),
        "language": language,
        "detection_model": detection_model,
        "recognition_model": recognition_model,
        "text_blocks": [
            {
                "id": i,
                "box": [int(x) for x in box],
                "text": text
            }
            for i, (box, text) in enumerate(ocr_results)
        ]
    }
    
    # If output path is provided, save results
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Determine output file name based on input image
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # Save results as JSON
        json_path = os.path.join(output_path, f"{base_name}_ocr.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved results to: {json_path}")
        
        # If requested, generate and save visualization
        if save_visualization:
            viz_image = image.copy()
            
            # Draw bounding boxes and text
            for block in results["text_blocks"]:
                x_min, y_min, x_max, y_max = block["box"]
                text = block["text"]
                
                # Draw rectangle
                cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(viz_image, f"Block {block['id']}", (x_min, y_min - 10),
                            font, 0.5, (0, 255, 0), 2)
                
                # Add text content (truncated if long)
                display_text = text[:25] + "..." if len(text) > 25 else text
                cv2.putText(viz_image, display_text, (x_min, y_max + 20),
                            font, 0.5, (0, 0, 255), 1)
            
            # Save visualization
            viz_path = os.path.join(output_path, f"{base_name}_ocr_viz.jpg")
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
    Process all images in a folder with OCR.
    
    Args:
        input_folder: Path to the folder containing images
        output_folder: Path to the output folder
        recursive: Whether to search for images recursively
        **kwargs: Additional arguments to pass to process_single_image
        
    Returns:
        Dictionary mapping image paths to their OCR results
    """
    logger.info(f"Processing folder: {input_folder}")
    
    # Create output folder if needed
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Find image files
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
    
    # Process each image
    results = {}
    for image_path in sorted(image_files):
        rel_path = os.path.relpath(image_path, input_folder)
        
        # Create corresponding output folder if recursive
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
    
    # Save summary if output folder is provided
    if output_folder:
        summary_path = os.path.join(output_folder, "ocr_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Create a simplified summary to avoid huge files
            summary = {
                k: {
                    "text_blocks_count": len(v.get("text_blocks", [])),
                    "error": v.get("error")
                }
                for k, v in results.items()
            }
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary to: {summary_path}")
    
    return results

def run_with_args():
    """Run the script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Process images with OCR for comic translation")
    
    # Input/output arguments
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or folder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output folder")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Process folders recursively")
    
    # OCR configuration
    parser.add_argument("--languages", "-l", type=str, default="ko,en",
                        help="Comma-separated list of language codes (e.g., 'ko,en')")
    parser.add_argument("--detection", "-d", type=str, default="craft",
                        choices=["craft", "east"],
                        help="Text detection model to use")
    parser.add_argument("--recognition", "-m", type=str, default="trocr",
                        choices=["trocr", "easyocr"],
                        help="Text recognition model to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom model weights")
    
    # Processing options
    parser.add_argument("--min-block-size", type=int, default=20,
                        help="Minimum size of text blocks to be considered")
    parser.add_argument("--paragraph-mode", action="store_true",
                        help="Group lines into paragraphs")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Don't save visualization images")
    
    args = parser.parse_args()
    
    # Convert comma-separated languages to list
    languages = args.languages.split(',')
    
    try:
        # Check if input is a file or directory
        if os.path.isfile(args.input):
            process_single_image(
                image_path=args.input,
                output_path=args.output,
                detection_model=args.detection,
                recognition_model=args.recognition,
                language=languages,
                custom_model_path=args.model_path,
                save_visualization=not args.no_visualization,
                min_block_size=args.min_block_size,
                paragraph_mode=args.paragraph_mode
            )
            logger.info("Successfully processed image")
            
        elif os.path.isdir(args.input):
            process_folder(
                input_folder=args.input,
                output_folder=args.output,
                recursive=args.recursive,
                detection_model=args.detection,
                recognition_model=args.recognition,
                language=languages,
                custom_model_path=args.model_path,
                save_visualization=not args.no_visualization,
                min_block_size=args.min_block_size,
                paragraph_mode=args.paragraph_mode
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
    # Load environment variables
    TROCR_MODEL_PATH = os.path.abspath(os.getenv("TROCR_MODEL_PATH", "./models/TrOCR/tuned"))
    INPUT_PATH = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/bubbles/"))
    OUTPUT_PATH = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    LANGUAGES_STR = os.getenv("LANGUAGE", "ko,en")
    LANGUAGES = LANGUAGES_STR.split(',')
    
    if os.path.isfile(INPUT_PATH):
        process_single_image(
            image_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            language=LANGUAGES,
            custom_model_path=TROCR_MODEL_PATH,
            save_visualization=True
        )
    else:
        process_folder(
            input_folder=INPUT_PATH,
            output_folder=OUTPUT_PATH,
            recursive=False,
            language=LANGUAGES,
            custom_model_path=TROCR_MODEL_PATH,
            save_visualization=True
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