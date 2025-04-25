#!/usr/bin/env python3
"""
Comic processing script for ComicGlot.

This script provides a command-line interface for the full ComicGlot pipeline,
including detection, recognition, and translation of text in comic images.

Example usage:
    python process_chapter.py --input path/to/image.jpg --output path/to/output
    python process_chapter.py --input path/to/chapter --output path/to/output --batch
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from comicglot.comic_processor import ComicProcessor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_single(
    input_path: str,
    output_path: str,
    detection_model: str,
    recognition_model: str,
    translator: str,
    source_lang: str,
    target_lang: str,
    detection_model_path: str,
    recognition_model_path: str,
    dual_ocr: bool,
    use_correction: bool,
    use_segmentation: bool,
    min_seg_height: int,
    save_segments: bool,
    save_detection: bool,
    save_recognition: bool,
    save_translation: bool,
    save_crops: bool
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
        dual_ocr: Whether to use dual OCR for better accuracy
        use_correction: Whether to use LLM-based OCR correction
        use_segmentation: Whether to use segmentation for long comics
        min_seg_height: Minimum height of segments when using segmentation
        save_segments: Whether to save segmented images
        save_detection: Whether to save detection visualizations
        save_recognition: Whether to save recognition visualizations
        save_translation: Whether to save translation visualizations
        save_crops: Whether to save cropped text blocks
        
    Returns:
        Processing results
    """
    processor = ComicProcessor(
        detection_model=detection_model,
        recognition_model=recognition_model,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path,
        dual_ocr=dual_ocr,
        use_correction=use_correction,
        use_segmentation=use_segmentation,
        min_seg_height=min_seg_height
    )
    
    if os.path.isfile(input_path):
        logger.info(f"Processing single image: {input_path}")
        results = processor.process_image(
            input_path, 
            output_path, 
            save_detection=save_detection,
            save_recognition=save_recognition,
            save_translation=save_translation
        )
    else:
        logger.info(f"Processing chapter: {input_path}")
        results = processor.process_chapter(
            input_path, 
            output_path, 
            save_segments=save_segments,
            save_detection=save_detection,
            save_recognition=save_recognition,
            save_translation=save_translation,
            save_crops=save_crops
        )
    
    return results

def process_batch(
    input_path: str,
    output_path: str,
    detection_model: str,
    recognition_model: str,
    translator: str,
    source_lang: str,
    target_lang: str,
    detection_model_path: str,
    recognition_model_path: str,
    dual_ocr: bool,
    use_correction: bool,
    use_segmentation: bool,
    min_seg_height: int,
    save_segments: bool,
    save_detection: bool,
    save_recognition: bool,
    save_translation: bool,
    save_crops: bool
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
        dual_ocr: Whether to use dual OCR for better accuracy
        use_correction: Whether to use LLM-based OCR correction
        use_segmentation: Whether to use segmentation for long comics
        min_seg_height: Minimum height of segments when using segmentation
        save_segments: Whether to save segmented images
        save_detection: Whether to save detection visualizations
        save_recognition: Whether to save recognition visualizations
        save_translation: Whether to save translation visualizations
        save_crops: Whether to save cropped text blocks
        
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
        recognition_model_path=recognition_model_path,
        dual_ocr=dual_ocr,
        use_correction=use_correction,
        use_segmentation=use_segmentation,
        min_seg_height=min_seg_height
    )
    
    logger.info(f"Processing chapters in batch mode: {input_path}")
    results = processor.batch_process_chapters(
        input_path, 
        output_path, 
        save_segments=save_segments,
        save_detection=save_detection,
        save_recognition=save_recognition,
        save_translation=save_translation,
        save_crops=save_crops
    )
    
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
    
    # Processing options
    parser.add_argument("--dual-ocr", action="store_true",
                        help="Use dual OCR for better accuracy")
    parser.add_argument("--correction", action="store_true",
                        help="Use LLM-based OCR correction")
    parser.add_argument("--segmentation", action="store_true",
                        help="Use segmentation for long vertical comics")
    parser.add_argument("--min-seg-height", type=int, default=800,
                        help="Minimum segment height when using segmentation")
    
    # Save options
    parser.add_argument("--save-segments", action="store_true",
                        help="Save segmented images to disk")
    parser.add_argument("--save-detection", action="store_true",
                        help="Save detection visualizations")
    parser.add_argument("--save-recognition", action="store_true",
                        help="Save recognition visualizations")
    parser.add_argument("--save-translation", action="store_true",
                        help="Save translation visualizations")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save cropped text blocks")
    parser.add_argument("--save-all", action="store_true",
                        help="Save all intermediate results")
    
    # Advanced options
    parser.add_argument("--detection-model-path", type=str, default=None,
                        help="Path to custom detection model")
    parser.add_argument("--recognition-model-path", type=str, default=None,
                        help="Path to custom recognition model")
    
    args = parser.parse_args()
    
    if args.save_all:
        args.save_segments = True
        args.save_detection = True
        args.save_recognition = True
        args.save_translation = True
        args.save_crops = True

    output_path = args.output
    if output_path:
        from comicglot.utils.file_utils import get_next_run_folder
        output_path = get_next_run_folder(output_path, "comicglot")
        logger.info(f"Results will be saved to: {output_path}")
    
    try:
        if args.batch:
            process_batch(
                input_path=args.input,
                output_path=output_path,
                detection_model=args.detection,
                recognition_model=args.recognition,
                translator=args.translator,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                detection_model_path=args.detection_model_path,
                recognition_model_path=args.recognition_model_path,
                dual_ocr=args.dual_ocr,
                use_correction=args.correction,
                use_segmentation=args.segmentation,
                min_seg_height=args.min_seg_height,
                save_segments=args.save_segments,
                save_detection=args.save_detection,
                save_recognition=args.save_recognition,
                save_translation=args.save_translation,
                save_crops=args.save_crops
            )
            logger.info("Successfully processed chapters in batch mode")
            
        else:
            process_single(
                input_path=args.input,
                output_path=output_path,
                detection_model=args.detection,
                recognition_model=args.recognition,
                translator=args.translator,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                detection_model_path=args.detection_model_path,
                recognition_model_path=args.recognition_model_path,
                dual_ocr=args.dual_ocr,
                use_correction=args.correction,
                use_segmentation=args.segmentation,
                min_seg_height=args.min_seg_height,
                save_segments=args.save_segments,
                save_detection=args.save_detection,
                save_recognition=args.save_recognition,
                save_translation=args.save_translation,
                save_crops=args.save_crops
            )
            logger.info("Successfully processed input")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_direct():
    """Run the script with hard-coded parameters from environment variables."""
    # Load parameters from environment variables
    input_path = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/chapters/"))
    output_path = os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    detection_model = os.getenv("DETECTION_MODEL", "yolo")
    recognition_model = os.getenv("RECOGNITION_MODEL", "trocr-korean")
    translator = os.getenv("TRANSLATOR")
    source_lang = os.getenv("SOURCE_LANG", "ko")
    target_lang = os.getenv("TARGET_LANG", "en")
    detection_model_path = os.getenv("YOLO_MODEL_PATH" if detection_model == "yolo" else "DETR_MODEL_PATH")
    recognition_model_path = os.getenv("TROCR_MODEL_PATH")
    dual_ocr = os.getenv("DUAL_OCR", "False").lower() in ("true", "1", "yes")
    use_correction = os.getenv("USE_CORRECTION", "False").lower() in ("true", "1", "yes")
    use_segmentation = os.getenv("USE_SEGMENTATION", "True").lower() in ("true", "1", "yes")
    min_seg_height = int(os.getenv("MIN_SEG_HEIGHT", "800"))
    
    # Save options
    save_segments = os.getenv("SAVE_SEGMENTS", "True").lower() in ("true", "1", "yes")
    save_detection = os.getenv("SAVE_DETECTION", "True").lower() in ("true", "1", "yes")
    save_recognition = os.getenv("SAVE_RECOGNITION", "True").lower() in ("true", "1", "yes")
    save_translation = os.getenv("SAVE_TRANSLATION", "True").lower() in ("true", "1", "yes")
    save_crops = os.getenv("SAVE_CROPS", "False").lower() in ("true", "1", "yes")
    
    # Create output directory for this run
    if output_path:
        from comicglot.utils.file_utils import get_next_run_folder
        output_path = get_next_run_folder(output_path, "comicglot")
        logger.info(f"Results will be saved to: {output_path}")

    is_batch_mode = False
    if os.path.isdir(input_path):
        subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        has_image_subdirs = False
        for subdir in subdirs:
            subdir_path = os.path.join(input_path, subdir)
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    has_image_subdirs = True
                    break
            if has_image_subdirs:
                break
        is_batch_mode = has_image_subdirs and len(subdirs) > 1
    
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
                dual_ocr=dual_ocr,
                use_correction=use_correction,
                use_segmentation=use_segmentation,
                min_seg_height=min_seg_height,
                save_segments=save_segments,
                save_detection=save_detection,
                save_recognition=save_recognition,
                save_translation=save_translation,
                save_crops=save_crops
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
                dual_ocr=dual_ocr,
                use_correction=use_correction,
                use_segmentation=use_segmentation,
                min_seg_height=min_seg_height,
                save_segments=save_segments,
                save_detection=save_detection,
                save_recognition=save_recognition,
                save_translation=save_translation,
                save_crops=save_crops
            )
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

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