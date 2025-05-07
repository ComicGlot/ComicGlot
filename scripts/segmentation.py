"""
Comic page segmentation pipeline.

This module provides high-level functions for segmenting comic/manhwa pages
using the seam carving algorithm. It can be used as an imported library
or run directly as a script.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Union, Tuple
import cv2
import numpy as np
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.utils.seam_segmentor import segment_by_seam_carving

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_chapter(
    chapter_path: str,
    output_path: Optional[str] = None,
    min_height: int = 800,
    save_crops: bool = True,
    resize: bool = True,
    tolerance: float = 0.9,
    carving_step: int = 20
) -> Dict:
    """
    Process a full chapter folder containing comic/manhwa images.
    
    Args:
        chapter_path: Path to folder containing chapter images
        output_path: Path to save the processed segments (if None, uses chapter_path/segments)
        min_height: Minimum height of each segment in pixels
        save_crops: Whether to save individual crops
        resize: If True, resizes all images to the same width
        tolerance: Threshold for color homogeneity (0.0-1.0)
        carving_step: Window size for finding local maxima of homogeneity
        
    Returns:
        Dict containing segmentation results and metadata
    """
    logging.info(f"Processing chapter: {chapter_path}")
    
    if output_path is None:
        output_path = os.path.join(chapter_path, "segments")

    if save_crops and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        logging.info(f"Created output directory: {output_path}")

    result = segment_by_seam_carving(
        input_path=chapter_path,
        min_height=min_height,
        save_crops=save_crops,
        output_path=output_path,
        resize=resize,
        tolerance=tolerance,
        carving_step=carving_step
    )

    result['processed_time'] = logging.Formatter.converter()
    result['num_segments'] = len(result['segments'])
    result['config'] = {
        'min_height': min_height,
        'resize': resize,
        'tolerance': tolerance,
        'carving_step': carving_step
    }
    
    logging.info(f"Chapter processing complete. Created {result['num_segments']} segments.")
    
    return result


def batch_process_chapters(
    base_path: str,
    output_base_path: Optional[str] = None,
    min_height: int = 800,
    save_crops: bool = True,
    resize: bool = True,
    tolerance: float = 0.9,
    carving_step: int = 20
) -> Dict[str, Dict]:
    """
    Process multiple chapter folders within a base directory.
    
    Args:
        base_path: Path containing multiple chapter folders
        output_base_path: Base path for outputs (if None, uses subdirectories in original locations)
        min_height: Minimum height of each segment in pixels
        save_crops: Whether to save individual crops
        resize: If True, resizes all images to the same width
        tolerance: Threshold for color homogeneity (0.0-1.0)
        carving_step: Window size for finding local maxima of homogeneity
        
    Returns:
        Dict mapping chapter names to their processing results
    """
    logging.info(f"Batch processing chapters in: {base_path}")

    chapter_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            chapter_folders.append(item)
    
    logging.info(f"Found {len(chapter_folders)} potential chapter folders")
    
    results = {}
    for chapter_name in natsorted(chapter_folders):
        chapter_path = os.path.join(base_path, chapter_name)

        if output_base_path:
            chapter_output_path = os.path.join(output_base_path, chapter_name)
        else:
            chapter_output_path = os.path.join(chapter_path, "segments")
        
        try:
            chapter_result = process_chapter(
                chapter_path=chapter_path,
                output_path=chapter_output_path,
                min_height=min_height,
                save_crops=save_crops,
                resize=resize,
                tolerance=tolerance,
                carving_step=carving_step
            )
            results[chapter_name] = chapter_result
            
        except Exception as e:
            logging.error(f"Error processing chapter {chapter_name}: {str(e)}")
            results[chapter_name] = {"error": str(e)}
    
    logging.info(f"Batch processing complete. Processed {len(results)} chapters.")
    
    return results


def visualize_segments(result: Dict, output_path: Optional[str] = None) -> None:
    """
    Visualize the segment boundaries on the composite image.
    
    Args:
        result: Result dictionary from segment_by_seam_carving
        output_path: Path to save the visualization (if None, doesn't save)
    """
    if 'segments' not in result or not result['segments']:
        logging.error("No segments found in the result")
        return

    segments = result['segments']
    boundaries = result['segment_boundaries']
    total_height = result['composite_height']
    total_width = result['composite_width']

    vis_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    for segment, (start_y, end_y) in zip(segments, boundaries):
        vis_image[start_y:end_y, :, :] = segment
    
    for _, (_, y) in enumerate(boundaries[:-1]):
        cv2.line(vis_image, (0, y), (total_width, y), (0, 0, 255), 2)

    if output_path:
        cv2.imwrite(output_path, vis_image)
        logging.info(f"Saved visualization to: {output_path}")
    
    return vis_image


def run_with_args():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Segment comic/manhwa images using seam carving")
    
    parser.add_argument("input_path", type=str, help="Path to input folder containing images")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output folder for crops")
    parser.add_argument("--min-height", "-m", type=int, default=800, help="Minimum segment height in pixels")
    parser.add_argument("--no-save", action="store_true", help="Don't save crop images")
    parser.add_argument("--no-resize", action="store_true", help="Don't resize images to match width")
    parser.add_argument("--tolerance", "-t", type=float, default=0.9, help="Homogeneity tolerance (0.0-1.0)")
    parser.add_argument("--carving-step", "-c", type=int, default=20, help="Window size for finding split points")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all subdirectories as chapters")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization of segments")
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            results = batch_process_chapters(
                base_path=args.input_path,
                output_base_path=args.output,
                min_height=args.min_height,
                save_crops=not args.no_save,
                resize=not args.no_resize,
                tolerance=args.tolerance,
                carving_step=args.carving_step
            )
            
            # Generate visualizations if requested
            if args.visualize:
                for chapter_name, result in results.items():
                    if 'error' in result:
                        continue
                    
                    if args.output:
                        vis_path = os.path.join(args.output, chapter_name, "visualization.png")
                    else:
                        vis_path = os.path.join(args.input_path, chapter_name, "visualization.png")
                    
                    visualize_segments(result, vis_path)
            
            logging.info(f"Successfully processed {len(results)} chapters")
            
        else:
            result = process_chapter(
                chapter_path=args.input_path,
                output_path=args.output,
                min_height=args.min_height,
                save_crops=not args.no_save,
                resize=not args.no_resize,
                tolerance=args.tolerance,
                carving_step=args.carving_step
            )
            
            # Generate visualization if requested
            if args.visualize:
                if args.output:
                    vis_path = os.path.join(args.output, "visualization.png")
                else:
                    vis_path = os.path.join(args.input_path, "visualization.png")
                
                visualize_segments(result, vis_path)
            
            logging.info(f"Successfully processed chapter with {result['num_segments']} segments")
            
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return 1
    
    return 0

def run_direct():
    """
    Run segmentation with hard-coded parameters.
    """
    INPUT_PATH = os.path.abspath(os.getenv("INPUT_FOLDER", "./data/chapters/"))
    OUTPUT_PATH = None # os.path.abspath(os.getenv("OUTPUT_FOLDER", "./data/output/"))
    MIN_HEIGHT = 800  # Minimum height of each segment in pixels
    SAVE_CROPS = True  # Whether to save individual crops
    RESIZE = True  # If True, resize all images to the same width
    TOLERANCE = 0.95  # Threshold for color homogeneity (0.0-1.0)
    CARVING_STEP = 10  # Window size for finding split points
    BATCH_MODE = False  # Process all subdirectories as chapters
    VISUALIZE = True  # Generate visualization of segments

    try:
        if BATCH_MODE:
            logging.info(f"Running in batch mode on: {INPUT_PATH}")
            results = batch_process_chapters(
                base_path=INPUT_PATH,
                output_base_path=OUTPUT_PATH,
                min_height=MIN_HEIGHT,
                save_crops=SAVE_CROPS,
                resize=RESIZE,
                tolerance=TOLERANCE,
                carving_step=CARVING_STEP
            )
            
            # Generate visualizations if requested
            if VISUALIZE:
                for chapter_name, result in results.items():
                    if 'error' in result:
                        continue
                    
                    if OUTPUT_PATH:
                        vis_path = os.path.join(OUTPUT_PATH, chapter_name, "visualization.png")
                    else:
                        vis_path = os.path.join(INPUT_PATH, chapter_name, "visualization.png")
                    
                    visualize_segments(result, vis_path)
            
            logging.info(f"Successfully processed {len(results)} chapters")
            
        else:
            logging.info(f"Processing single chapter: {INPUT_PATH}")
            result = process_chapter(
                chapter_path=INPUT_PATH,
                output_path=OUTPUT_PATH,
                min_height=MIN_HEIGHT,
                save_crops=SAVE_CROPS,
                resize=RESIZE,
                tolerance=TOLERANCE,
                carving_step=CARVING_STEP
            )
            
            # Generate visualization if requested
            if VISUALIZE:
                if OUTPUT_PATH:
                    vis_path = os.path.join(OUTPUT_PATH, "visualization.png")
                else:
                    vis_path = os.path.join(INPUT_PATH, "visualization.png")
                
                visualize_segments(result, vis_path)
            
            logging.info(f"Successfully processed chapter with {result['num_segments']} segments")
            
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
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