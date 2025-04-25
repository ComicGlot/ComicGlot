"""
OCR recognition script for comic processing.

This script provides a standalone interface for OCR processing of text bubbles
in comic images. It can process individual images or folders of cropped text bubbles,
perform CRAFT text line detection, and recognize text using TrOCR or EasyOCR.
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
from natsort import natsorted

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.recognition import OCRProcessor
from comicglot.utils.file_utils import load_image, save_image, get_next_run_folder
from comicglot.utils.visualization import draw_boxes_with_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ocr_recognition")
load_dotenv()

def process_single_image(
    image_path: str,
    ocr_processor: OCRProcessor,
    output_path: Optional[str] = None,
    min_block_size: int = 20,
    use_line_detection: bool = True,
    paragraph_mode: bool = False,
    save_visualization: bool = True,
    save_line_crops: bool = False
) -> Dict:
    """
    Process a single image for OCR.
    
    Args:
        image_path: Path to the input image
        ocr_processor: Initialized OCRProcessor instance
        output_path: Base path for output files
        min_block_size: Minimum size of text blocks to detect
        use_line_detection: Whether to use text line detection
        paragraph_mode: Whether to group lines into paragraphs
        save_visualization: Whether to save visualization images
        save_line_crops: Whether to save cropped text lines
        
    Returns:
        Dictionary containing OCR results and metadata
    """
    image = load_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    if use_line_detection:
        ocr_results = ocr_processor.perform_detection_and_recognition(
            image, min_block_size=min_block_size, 
            paragraph_mode=paragraph_mode
        )
    else:
        ocr_results = ocr_processor.perform_detection_and_recognition(
            image, detection=False
        )

    output_text = " ".join([text for _, text in ocr_results])
    
    results = {
        "image_path": image_path,
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": output_text,
        "line_count": len(ocr_results),
        "lines": [
            {
                "box": [int(x) for x in box],
                "text": text
            }
            for box, text in ocr_results
        ]
    }

    if output_path and (save_visualization or save_line_crops):
        os.makedirs(output_path, exist_ok=True)
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]

        if save_visualization and len(ocr_results) > 0:
            vis_image = image.copy()
            vis_dir = os.path.join(output_path, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            boxes = [np.array(box) for box, _ in ocr_results]
            classes = ["text_line"] * len(ocr_results)
            confidences = [1.0] * len(ocr_results)  # No confidence scores available
            ids = list(range(len(ocr_results)))
            texts = [text for _, text in ocr_results]
            
            vis_image = draw_boxes_with_labels(vis_image, boxes, texts)
            
            viz_path = os.path.join(vis_dir, f"{base_name}_ocr.jpg")
            save_image(vis_image, viz_path)

        if save_line_crops and len(ocr_results) > 0:
            lines_dir = os.path.join(output_path, "lines")
            os.makedirs(lines_dir, exist_ok=True)
            
            for i, (box, text) in enumerate(ocr_results):
                x_min, y_min, x_max, y_max = map(int, box)
                crop = image[y_min:y_max, x_min:x_max].copy()
                
                # Use standardized naming convention
                crop_path = os.path.join(lines_dir, f"{base_name}_l{i+1:03d}.jpg")
                save_image(crop, crop_path)
    
    return results

def process_folder(
    input_folder: str,
    ocr_processor: OCRProcessor,
    output_folder: Optional[str] = None,
    recursive: bool = False,
    min_block_size: int = 20,
    use_line_detection: bool = True,
    paragraph_mode: bool = False,
    save_visualization: bool = True,
    save_line_crops: bool = False
) -> Dict:
    """
    Process all images in a folder for OCR.
    
    Args:
        input_folder: Path to the folder containing images
        ocr_processor: Initialized OCRProcessor instance
        output_folder: Path to the output folder
        recursive: Whether to search for images recursively
        min_block_size: Minimum size of text blocks to detect
        use_line_detection: Whether to use text line detection
        paragraph_mode: Whether to group lines into paragraphs
        save_visualization: Whether to save visualization images
        save_line_crops: Whether to save cropped text lines
        
    Returns:
        Dictionary with aggregated OCR results
    """
    logger.info(f"Processing folder: {input_folder}")

    if output_folder:
        run_folder = get_next_run_folder(output_folder, "recognition")
        logger.info(f"Saving outputs to: {run_folder}")
    else:
        run_folder = None

    if save_visualization and run_folder:
        viz_dir = os.path.join(run_folder, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
    
    if save_line_crops and run_folder:
        lines_dir = os.path.join(run_folder, "lines")
        os.makedirs(lines_dir, exist_ok=True)

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

    image_files = natsorted(image_files)

    all_results = {
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(image_files),
        "total_lines": 0,
        "total_characters": 0,
        "results": {}
    }

    for image_path in image_files:
        rel_path = os.path.relpath(image_path, input_folder)
        
        try:
            image_results = process_single_image(
                image_path=image_path,
                ocr_processor=ocr_processor,
                output_path=run_folder,
                min_block_size=min_block_size,
                use_line_detection=use_line_detection,
                paragraph_mode=paragraph_mode,
                save_visualization=save_visualization,
                save_line_crops=save_line_crops
            )
            
            all_results["total_lines"] += len(image_results["lines"])
            all_results["total_characters"] += len(image_results["text"])

            all_results["results"][rel_path] = {
                "text": image_results["text"],
                "line_count": len(image_results["lines"])
            }
            
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {str(e)}")
            all_results["results"][rel_path] = {"error": str(e)}

    if run_folder:
        summary_path = os.path.join(run_folder, "ocr_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        all_text_path = os.path.join(run_folder, "all_text.txt")
        with open(all_text_path, 'w', encoding='utf-8') as f:
            for path, result in sorted(all_results["results"].items()):
                if "text" in result:
                    f.write(f"# {path}\n{result['text']}\n\n")
        
        logger.info(f"Recognized {all_results['total_lines']} text lines with {all_results['total_characters']} characters across {len(image_files)} images")
    
    return all_results

def run_with_args():
    """Run the script with command-line arguments."""
    parser = argparse.ArgumentParser(description="OCR processing for comic text bubbles")
    
    # Input/output arguments
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or folder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output folder")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Process folders recursively")
    
    # OCR configuration
    parser.add_argument("--language", "-l", type=str, default="ko,en",
                        help="Comma-separated language codes (e.g., 'ko,en')")
    parser.add_argument("--detection", "-d", type=str, default="craft",
                        choices=["craft", "east", "none"],
                        help="Text line detection model")
    parser.add_argument("--recognition", "-rec", type=str, default="trocr",
                        choices=["trocr", "easyocr"],
                        help="Text recognition model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to custom OCR model")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    
    # Processing options
    parser.add_argument("--min-block-size", type=int, default=20,
                        help="Minimum size of text blocks to detect")
    parser.add_argument("--no-line-detection", action="store_true",
                        help="Disable text line detection (process whole image)")
    parser.add_argument("--paragraph-mode", action="store_true",
                        help="Group text lines into paragraphs")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Don't save visualization images")
    parser.add_argument("--save-line-crops", action="store_true",
                        help="Save cropped text lines")
    
    args = parser.parse_args()
    languages = args.language.split(',')
    
    try:
        ocr_processor = OCRProcessor(
            language=languages,
            detection_model=args.detection if args.detection != "none" else "craft",
            recognition_model=args.recognition,
            custom_model_path=args.model_path,
            use_gpu=not args.no_gpu
        )
        
        if os.path.isfile(args.input):
            if args.output:
                run_folder = get_next_run_folder(args.output, "recognition")
                logger.info(f"Saving outputs to: {run_folder}")
            else:
                run_folder = None
            
            results = process_single_image(
                image_path=args.input,
                ocr_processor=ocr_processor,
                output_path=run_folder,
                min_block_size=args.min_block_size,
                use_line_detection=not args.no_line_detection,
                paragraph_mode=args.paragraph_mode,
                save_visualization=not args.no_visualization,
                save_line_crops=args.save_line_crops
            )

            if run_folder:
                results_path = os.path.join(run_folder, "ocr_results.json")
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                text_path = os.path.join(run_folder, "text.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(results["text"])
            
            logger.info(f"Recognized text: {results['text']}")
            
        elif os.path.isdir(args.input):
            process_folder(
                input_folder=args.input,
                ocr_processor=ocr_processor,
                output_folder=args.output,
                recursive=args.recursive,
                min_block_size=args.min_block_size,
                use_line_detection=not args.no_line_detection,
                paragraph_mode=args.paragraph_mode,
                save_visualization=not args.no_visualization,
                save_line_crops=args.save_line_crops
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
    # Load parameters from environment variables
    TROCR_MODEL_PATH = os.path.abspath(os.getenv("TROCR_MODEL_PATH", "./models/KoTrOCR/checkpoint-12000-v6"))
    # INPUT_BUBBLE = os.path.abspath(os.getenv("INPUT_BUBBLE", "./data/bubbles/test.jpg"))
    INPUT_BUBBLE_FOLDER = os.path.abspath(os.getenv("INPUT_BUBBLE_FOLDER", "./data/bubbles/"))
    OUTPUT_FOLDER = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    LANGUAGES_STR = os.getenv("LANGUAGE", "ko,en")
    LANGUAGES = LANGUAGES_STR.split(',')
    
    logger.info(f"Starting OCR processing")
    
    # Determine whether to process single image or folder
    input_path = INPUT_BUBBLE_FOLDER if INPUT_BUBBLE_FOLDER else INPUT_BUBBLE
    logger.info(f"Input: {input_path}, Output: {OUTPUT_FOLDER}")
    
    try:
        ocr_processor = OCRProcessor(
            language=LANGUAGES,
            detection_model="craft",
            recognition_model="trocr",
            custom_model_path=TROCR_MODEL_PATH,
            use_gpu=True
        )

        if os.path.isfile(input_path):
            run_folder = get_next_run_folder(OUTPUT_FOLDER, "recognition")
            logger.info(f"Saving outputs to: {run_folder}")

            results = process_single_image(
                image_path=input_path,
                ocr_processor=ocr_processor,
                output_path=run_folder,
                min_block_size=20,
                use_line_detection=True,
                paragraph_mode=False,
                save_visualization=True,
                save_line_crops=True
            )

            results_path = os.path.join(run_folder, "ocr_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            text_path = os.path.join(run_folder, "text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(results["text"])
            
            logger.info(f"Recognized {len(results['lines'])} text lines")
            
        elif os.path.isdir(input_path):
            process_folder(
                input_folder=input_path,
                ocr_processor=ocr_processor,
                output_folder=OUTPUT_FOLDER,
                recursive=False,
                min_block_size=20,
                use_line_detection=True,
                paragraph_mode=False,
                save_visualization=True,
                save_line_crops=True
            )
            
        else:
            logger.error(f"Input path does not exist: {input_path}")
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

# SORTING ALGO
# 1. CHANGE SORTING INSIDE GROUP TO MORE COMPLEX. CREATE SUBGROUPS FOR WHEN Y1_OTHER IS IN RANGE Y1 to Y1.5
#    THEN SORT BY X. IF NOT - STILL SORT BY Y. BUT WE SEARCH FOR OVERLAPS IN WHOLE GROUP, THAT`S WHAT IT`S FOR
# 2. CHANGE CUTING. IF CUT IS EXACTLY ON X OR Y COORDINATE - MEANS ONE BLOCK IS SMALLER AND CUT IN HALF. 
#    SO WE CHANGE THE ALGO HERE. DO NOT CUT ANYTHING FROM SMALLER ONE, AND FULLY MASK THE OVERLAPING PART ON THE BIG BUBBLE.