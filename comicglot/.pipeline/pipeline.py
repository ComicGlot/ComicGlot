"""
ComicGlot core pipeline module.

This module provides the main processing pipeline for ComicGlot,
integrating segmentation, detection, recognition, and translation
components into a cohesive workflow.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
import numpy as np
import time
from natsort import natsorted

class ComicProcessor:
    """
    Main processing pipeline for ComicGlot.
    """
    
    def __init__(
        self,
        detection_model: str = "yolo",
        recognition_model: str = "trocr-korean",
        translator: Optional[str] = None,
        source_lang: str = "ko",
        target_lang: str = "en",
        detection_model_path: Optional[str] = None,
        recognition_model_path: Optional[str] = None,
        apply_separators: bool = True
    ):
        """
        Initialize the comic processor.
        
        Args:
            detection_model: Model to use for text block detection ('yolo' or 'detr')
            recognition_model: Model to use for text recognition ('trocr-korean', 'trocr-base', 'easyocr')
            translator: Translation service to use (None, 'claude', 'flant5', etc.)
            source_lang: Source language code
            target_lang: Target language code
            detection_model_path: Custom path to detection model
            recognition_model_path: Custom path to recognition model
            apply_separators: Whether to apply separator masks to overlapping text blocks
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration
        self.detection_model_type = detection_model
        self.recognition_model_type = recognition_model
        self.translator_type = translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.apply_separators = apply_separators

        self.detector = self._init_detector(detection_model, detection_model_path)
        self.recognizer = self._init_recognizer(
            recognition_model, recognition_model_path, source_lang
        )
        self.translator = self._init_translator(translator, source_lang, target_lang)
        
        self.logger.info("ComicGlot processor initialized successfully")
    
    def _init_detector(self, model_type: str, model_path: Optional[str] = None):
        """
        Initialize the text block detector.
        
        Args:
            model_type: Type of detection model ('yolo' or 'detr')
            model_path: Path to custom model weights
            
        Returns:
            Initialized detector object
        """
        self.logger.info(f"Initializing {model_type} detector")
        
        if model_type.lower() == "yolo":
            from comicglot.detection import YOLODetector
            
            if model_path is None:
                # Try to get from environment
                import os
                from dotenv import load_dotenv
                load_dotenv()
                model_path = os.path.abspath(os.getenv("YOLO_MODEL_PATH", "./models/YOLO/best.pt"))
            
            return YOLODetector(model_path)
            
        elif model_type.lower() == "detr":
            from comicglot.detection import DETRDetector
            
            if model_path is None:
                # Try to get from environment
                import os
                from dotenv import load_dotenv
                load_dotenv()
                model_path = os.path.abspath(os.getenv("DETR_MODEL_PATH", "./models/DETR-RESNET-50"))
            
            return DETRDetector(model_path)
            
        else:
            self.logger.error(f"Unknown detection model: {model_type}")
            raise ValueError(f"Unknown detection model: {model_type}")
    
    def _init_recognizer(
        self, 
        model_type: str, 
        model_path: Optional[str] = None,
        language: str = "ko"
    ):
        """
        Initialize the text recognizer.
        
        Args:
            model_type: Type of recognition model
            model_path: Path to custom model weights
            language: Language code for OCR
            
        Returns:
            Initialized recognizer object
        """
        self.logger.info(f"Initializing {model_type} recognizer")
        
        from comicglot.recognition import OCRProcessor

        languages = [language]
        if language not in ["en"]:
            languages.append("en")  # Add English as fallback

        if model_type.lower() == "trocr-korean":
            detection_model = "craft"
            recognition_model = "trocr"
            
            if model_path is None:
                # Try to get from environment
                import os
                from dotenv import load_dotenv
                load_dotenv()
                model_path = os.path.abspath(os.getenv(
                    "TROCR_MODEL_PATH", 
                    "./models/TrOCR/tuned"
                ))
            
        elif model_type.lower() == "trocr-base":
            detection_model = "craft"
            recognition_model = "trocr"
            model_path = None  # Use default
            
        elif model_type.lower() == "easyocr":
            detection_model = "craft"
            recognition_model = "easyocr"
            model_path = None
            
        else:
            self.logger.error(f"Unknown recognition model: {model_type}")
            raise ValueError(f"Unknown recognition model: {model_type}")
        
        return OCRProcessor(
            language=languages,
            detection_model=detection_model,
            recognition_model=recognition_model,
            custom_model_path=model_path
        )
    
    def _init_translator(
        self,
        translator_type: Optional[str],
        source_lang: str,
        target_lang: str
    ):
        """
        Initialize the translator.
        
        Args:
            translator_type: Type of translator to use
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Initialized translator object or None if not specified
        """
        if translator_type is None:
            return None
            
        self.logger.info(f"Initializing {translator_type} translator")
        
        # Will implement this later
        self.logger.warning("Translation module will be implemented in a future update")
        return None
    
    def process_image(
        self, 
        image_path: str,
        output_path: Optional[str] = None,
        save_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            output_path: Directory to save output files (None for no saving)
            save_intermediates: Whether to save intermediate results
            
        Returns:
            Dictionary containing all processing results
        """
        self.logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        
        # Step 1: Detect text blocks
        self.logger.info("Step 1: Detecting text blocks")
        detect_time = time.time()
        detected_blocks, detection_viz = self.detector.process_image(
            image_path, return_visualization=save_intermediates
        )
        
        # Step 2: Apply spatial processing to rank blocks and identify separators
        self.logger.info("Step 2: Applying spatial processing")
        from comicglot.utils.spatial_processor import process_text_blocks
        from comicglot.utils.data_types import TextBlock

        text_blocks = [
            TextBlock(
                box=np.array(block["box"]),
                class_name=block["class_name"],
                confidence=block["confidence"],
                id=block["id"]
            )
            for block in detected_blocks
        ]

        spatial_time = time.time()
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
        
        # Step 3: Recognize text in each block
        self.logger.info("Step 3: Recognizing text in blocks")
        ocr_time = time.time()
        ocr_results = self.recognizer.process_image_with_blocks(
            image_path, processed_blocks, apply_separators=self.apply_separators
        )
        
        # Step 4: Translate text
        # self.logger.info("Step 4: Translating text")
        # translate_time = time.time()
        # if self.translator:
        #     # Translation will be implemented in future updates
        #     translated_results = ocr_results
        # else:
        #     translated_results = ocr_results
        
        # Calculate timing
        end_time = time.time()
        timing = {
            "detection": spatial_time - detect_time,
            "spatial_processing": ocr_time - spatial_time,
            "recognition": translate_time - ocr_time,
            "translation": end_time - translate_time if self.translator else 0,
            "total": end_time - start_time
        }
        
        # Prepare final results
        results = {
            "image_path": image_path,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timing": timing,
            "blocks": translated_results,
            "config": {
                "detection_model": self.detection_model_type,
                "recognition_model": self.recognition_model_type,
                "translator": self.translator_type,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "apply_separators": self.apply_separators
            }
        }

        if output_path:
            self._save_results(results, output_path, save_intermediates, detection_viz)
        
        return results
    
    def _save_results(
        self, 
        results: Dict[str, Any], 
        output_path: str,
        save_intermediates: bool,
        detection_viz: Optional[np.ndarray] = None
    ):
        """
        Save processing results to disk.
        
        Args:
            results: Processing results
            output_path: Directory to save output files
            save_intermediates: Whether to save intermediate results
            detection_viz: Detection visualization image
        """
        os.makedirs(output_path, exist_ok=True)

        image_name = os.path.basename(results["image_path"])
        base_name = os.path.splitext(image_name)[0]

        json_path = os.path.join(output_path, f"{base_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved results to {json_path}")
        
        # Save visualizations if requested
        if save_intermediates:
            if detection_viz is not None:
                viz_path = os.path.join(output_path, f"{base_name}_detection.jpg")
                cv2.imwrite(viz_path, detection_viz)
                self.logger.info(f"Saved detection visualization to {viz_path}")
            try:
                from comicglot.utils.visualization import draw_boxes_with_labels
                from comicglot.utils.file_utils import load_image
                
                image = load_image(results["image_path"])

                boxes = [np.array(block["box"]) for block in results["blocks"]]
                classes = [block["class_name"] for block in results["blocks"]]
                confidences = [block["confidence"] for block in results["blocks"]]
                ids = [block["id"] for block in results["blocks"]]

                labels = [f"{i+1}: {block.get('text', '')[:20]}" for i, block in enumerate(results["blocks"])]
                
                viz = draw_boxes_with_labels(image, boxes, labels)

                final_viz_path = os.path.join(output_path, f"{base_name}_final.jpg")
                cv2.imwrite(final_viz_path, viz)
                self.logger.info(f"Saved final visualization to {final_viz_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create final visualization: {e}")
    
    def process_chapter(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        save_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process all images in a chapter directory.
        
        Args:
            input_dir: Directory containing chapter images
            output_dir: Directory to save output files (None for no saving)
            save_intermediates: Whether to save intermediate results
            
        Returns:
            Dictionary containing all chapter processing results
        """
        self.logger.info(f"Processing chapter: {input_dir}")

        if not os.path.isdir(input_dir):
            self.logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        images_to_process = []
        
        if use_segmentation:
            self.logger.info("Using segmentation to split long comics")
            try:
                from comicglot.pipeline.segmentation import process_chapter

                segments_dir = None
                if output_dir and save_intermediates:
                    segments_dir = os.path.join(output_dir, "segments")

                segment_results = process_chapter(
                    input_dir, 
                    segments_dir, 
                    min_height=seg_min_height,
                    save_crops=save_intermediates
                )
                images_to_process.extend(segment_results['segments'])
                self.logger.info(f"Created {len(images_to_process)} segments")
                
            except ImportError as e:
                self.logger.warning(f"Segmentation module not available: {e}")
                self.logger.warning("Falling back to processing original images")
                use_segmentation = False
        
        if not use_segmentation:
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            image_files = natsorted(image_files)  # Ensure consistent order
            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file)
                images_to_process.append(image_path)
            
            self.logger.info(f"Found {len(images_to_process)} images to process")

        results = {"frames": []}
        
        for i, image in enumerate(images_to_process):
            self.logger.info(f"Processing frame {i+1}/{len(images_to_process)}")
            
            try:
                frame_output = None
                if output_dir:
                    frame_output = os.path.join(output_dir, f"frame_{i+1:03d}")
                
                # If image is a path string, process directly
                # If image is numpy array, save temporarily if needed
                image_path = image
                temp_file = None
                
                if isinstance(image, np.ndarray):
                    if output_dir and save_intermediates:
                        temp_file = os.path.join(output_dir, f"segment_{i+1:03d}.png")
                        cv2.imwrite(temp_file, image)
                        image_path = temp_file
                    else:
                        image_path = image

                frame_result = self.process_image(
                    image_path, frame_output, save_intermediates
                )

                results["frames"].append({
                    "frame_number": i + 1,
                    "blocks": frame_result["blocks"]
                })
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i+1}: {e}")
                results["frames"].append({
                    "frame_number": i + 1,
                    "error": str(e)
                })

        results["chapter_path"] = input_dir
        results["processed_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results["num_frames"] = len(images_to_process)
        results["num_blocks"] = sum(len(frame.get("blocks", [])) for frame in results["frames"] if "blocks" in frame)

        if output_dir:
            summary_path = os.path.join(output_dir, "chapter_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved chapter summary to {summary_path}")
        
        return results
    
    def batch_process_chapters(
        self,
        base_dir: str,
        output_base_dir: Optional[str] = None,
        save_intermediates: bool = False,
        use_segmentation: bool = False,
        seg_min_height: int = 800
    ) -> Dict[str, Any]:
        """
        Process multiple chapters in a base directory.
        
        Args:
            base_dir: Base directory containing chapter folders
            output_base_dir: Base directory for output files
            save_intermediates: Whether to save intermediate results
            use_segmentation: Whether to apply segmentation to split long comics
            seg_min_height: Minimum height for segmentation (only used if use_segmentation=True)
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"Batch processing chapters in: {base_dir}")

        chapter_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                chapter_dirs.append(item)
        
        chapter_dirs.sort()
        self.logger.info(f"Found {len(chapter_dirs)} potential chapter directories")

        results = {"chapters": {}}
        for chapter_name in chapter_dirs:
            chapter_path = os.path.join(base_dir, chapter_name)

            chapter_output = None
            if output_base_dir:
                chapter_output = os.path.join(output_base_dir, chapter_name)
                os.makedirs(chapter_output, exist_ok=True)
            
            try:
                self.logger.info(f"Processing chapter: {chapter_name}")
                chapter_result = self.process_chapter(
                    chapter_path, 
                    chapter_output, 
                    save_intermediates,
                    use_segmentation,
                    seg_min_height
                )
                results["chapters"][chapter_name] = {
                    "path": chapter_path,
                    "frames": len(chapter_result["frames"]),
                    "blocks": chapter_result["num_blocks"],
                    "status": "success"
                }
                self.logger.info(f"Successfully processed chapter: {chapter_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing chapter {chapter_name}: {str(e)}")
                results["chapters"][chapter_name] = {
                    "path": chapter_path,
                    "status": "error",
                    "error": str(e)
                }

        results["base_dir"] = base_dir
        results["processed_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results["num_chapters"] = len(chapter_dirs)
        results["successful_chapters"] = sum(
            1 for chapter in results["chapters"].values() 
            if chapter["status"] == "success"
        )

        if output_base_dir:
            summary_path = os.path.join(output_base_dir, "batch_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved batch summary to {summary_path}")
        
        return results