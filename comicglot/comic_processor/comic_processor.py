"""
ComicGlot core pipeline module.

This module provides the main processing pipeline for ComicGlot,
integrating segmentation, detection, recognition, and translation
components into a cohesive workflow.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
import numpy as np
from pathlib import Path
from natsort import natsorted

from comicglot.detection import YOLODetector, DETRDetector
from comicglot.recognition import OCRProcessor
from comicglot.translation import OCRCorrector, StyleTranslator

from comicglot.utils.data_types import TextBlock
from comicglot.utils.spatial_processor import process_text_blocks
from comicglot.utils.visualization import draw_boxes_with_labels, visualize_separator_lines
from comicglot.utils.file_utils import load_image, save_image, get_next_run_folder
from comicglot.utils.seam_segmentor import segment_by_seam_carving


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
        apply_separators: bool = True,
        dual_ocr: bool = False,
        use_correction: bool = False,
        use_segmentation: bool = False,
        min_seg_height: int = 800,
        correction_context: bool = True,
        class_condition: str = "true",
        context_window: int = 3,
        arbitrate_punctuation: bool = False
    ):
        """
        Initialize the comic processor.
        
        Args:
            detection_model: Model to use for text block detection ('yolo' or 'detr')
            recognition_model: Model to use for text recognition ('trocr-korean', 'trocr-base', 'easyocr')
            translator: Translation service to use (None, 'claude', etc.)
            source_lang: Source language code
            target_lang: Target language code
            detection_model_path: Custom path to detection model
            recognition_model_path: Custom path to recognition model
            apply_separators: Whether to apply separator masks to overlapping text blocks
            dual_ocr: Whether to use dual OCR for better accuracy
            use_correction: Whether to use LLM-based OCR correction
            use_segmentation: Whether to use segmentation for long vertical comics
            min_seg_height: Minimum height of segments when using segmentation
            correction_context: Give the OCR arbiter the block class and the previous lines
            class_condition: Style guide used for translation: "true" (the detected class),
                "none" (the default guide for every block) or "shuffled" (a deliberately
                mismatched class, for ablations); see StyleTranslator.WRONG_CLASS
            context_window: Number of previous blocks given to the translator (0 = none)
            arbitrate_punctuation: Send readings that differ only in , . ; : ! ? to the
                arbiter too. Off by default (the published behaviour); turn on when tone
                punctuation matters, for example for scream_bubble and think_bubble
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration
        self.detection_model_type = detection_model
        self.recognition_model_type = recognition_model
        self.translator_type = translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.apply_separators = apply_separators
        self.dual_ocr = dual_ocr
        self.use_correction = use_correction
        self.use_segmentation = use_segmentation
        self.min_seg_height = min_seg_height
        self.correction_context = correction_context
        if class_condition not in ("true", "none", "shuffled"):
            raise ValueError("class_condition must be 'true', 'none' or 'shuffled'")
        self.class_condition = class_condition
        self.context_window = context_window
        self.arbitrate_punctuation = arbitrate_punctuation
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path

        # Initialize components
        self.detector = self._init_detector(detection_model, detection_model_path)
        
        self.recognizer = self._init_recognizer(
            recognition_model, recognition_model_path, source_lang
        )
        
        if self.dual_ocr:
            self.secondary_recognizer = self._init_recognizer(
                "trocr-base" if recognition_model != "trocr-base" else "easyocr", 
                None, 
                source_lang
            )
        else:
            self.secondary_recognizer = None
            
        if self.use_correction:
            self.corrector = OCRCorrector()
        else:
            self.corrector = None
            
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
                    "./models/KoTrOCR/checkpoint-12000-v6"
                ))
            
        elif model_type.lower() == "trocr-base":
            detection_model = "craft"
            recognition_model = "trocr"
            model_path = None
            
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
        
        if translator_type.lower() == "claude":
            from comicglot.translation.style_processor import StyleTranslator
            return StyleTranslator(model_name="translation")
        else:
            self.logger.warning(f"Unknown translator type: {translator_type}")
            return None
    
    def process_image(
        self, 
        image_path: str,
        output_path: Optional[str] = None,
        save_detection: bool = False,
        save_recognition: bool = False,
        save_translation: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            output_path: Directory to save output files (None for no saving)
            save_detection: Whether to save detection visualization
            save_recognition: Whether to save recognition visualization
            save_translation: Whether to save translation visualization
            
        Returns:
            Dictionary containing all processing results
        """
        self.logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        
        # Step 1: Detect text blocks
        self.logger.info("Detecting text blocks")
        detect_time = time.time()
        detected_blocks = self.detector.process_image(
            image_path, conf=0.5, iou=0.7
        )
        
        # Step 2: Apply spatial processing to rank blocks and identify separators
        self.logger.info("Applying spatial processing")
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
        self.logger.info("Recognizing text in blocks")
        ocr_time = time.time()
        ocr_results = self.recognizer.process_image_with_blocks(
            image_path, processed_blocks, apply_separators=self.apply_separators
        )

        if self.dual_ocr and self.secondary_recognizer:
            self.logger.info("Running secondary OCR for comparison")
            secondary_ocr_results = self.secondary_recognizer.process_image_with_blocks(
                image_path, processed_blocks, apply_separators=self.apply_separators
            )

            for i, (primary_result, secondary_result) in enumerate(zip(ocr_results, secondary_ocr_results)):
                ocr_results[i]["secondary_text"] = secondary_result["text"]
        
        correction_time = time.time()
        translate_time = time.time()

        timing = {
            "detection": spatial_time - detect_time,
            "spatial_processing": ocr_time - spatial_time,
            "recognition": correction_time - ocr_time,
            "total": translate_time - start_time
        }

        results = {
            "image_path": image_path,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timing": timing,
            "blocks": ocr_results,
            "config": {
                "detection_model": self.detection_model_type,
                "recognition_model": self.recognition_model_type,
                "translator": self.translator_type,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "apply_separators": self.apply_separators,
                "dual_ocr": self.dual_ocr,
                "use_correction": self.use_correction
            }
        }

        if output_path:
            self._save_image_results(
                results, 
                image_path, 
                output_path, 
                save_detection=save_detection,
                save_recognition=save_recognition,
                save_translation=save_translation
            )
        
        return results
    
    def _provenance(self) -> Dict[str, Any]:
        """Everything needed to say which models and prompts produced a run."""
        import hashlib
        import os
        import subprocess
        def sha(path):
            try:
                with open(path, "rb") as fh:
                    return hashlib.sha256(fh.read()).hexdigest()[:16]
            except OSError:
                return None
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    timeout=5).stdout.strip() or None
        except Exception:
            commit = None
        det_path = self.detection_model_path or os.getenv("YOLO_MODEL_PATH")
        llm = {}
        if getattr(self, "corrector", None) is not None:
            llm["correction"] = {"model": self.corrector.llm_processor.model_name, "temperature": self.corrector.temperature}
        if getattr(self, "translator", None) is not None and hasattr(self.translator, "llm_processor"):
            llm["translation"] = {"model": self.translator.llm_processor.model_name, "temperature": self.translator.temperature}
        return {
            "git_commit": commit,
            "detector_weights": det_path, "detector_sha256_16": sha(det_path) if det_path else None,
            "recognition_model_path": self.recognition_model_path or os.getenv("TROCR_MODEL_PATH"),
            "llm": llm,
            "style_guides_sha256_16": sha(os.path.join("data", "prompts", "style_guides.json")),
            "correction_prompt_sha256_16": sha(os.path.join("data", "prompts", "correction_prompt.txt")),
        }

    def _are_texts_equivalent(self, text1: str, text2: str) -> bool:
        """
        Check if two OCR texts are equivalent, ignoring minor differences.
        
        Args:
            text1: First OCR text
            text2: Second OCR text
            
        Returns:
            True if texts are equivalent, False otherwise
        """
        if not text1 and not text2:
            return True
            
        if not text1 or not text2:
            return False

        def normalize(text):
            text = text.strip()
            text = ' '.join(text.split())
            text = text.replace('\n', ' ').replace('\r', ' ')
            if not self.arbitrate_punctuation:
                for char in ',.;:!?':
                    text = text.replace(char, '')
            return text.lower()
            
        return normalize(text1) == normalize(text2)
        
    def _save_image_results(
        self, 
        results: Dict[str, Any], 
        image_path: str,
        output_path: str,
        save_detection: bool = False,
        save_recognition: bool = False,
        save_translation: bool = False
    ):
        """
        Save processing results for a single image to disk.
        
        Args:
            results: Processing results
            image_path: Path to the original image
            output_path: Directory to save output files
            save_detection: Whether to save detection visualization
            save_recognition: Whether to save recognition visualization
            save_translation: Whether to save translation visualization
        """
        os.makedirs(output_path, exist_ok=True)

        image_name = os.path.basename(image_path) if isinstance(image_path, str) else "segment.jpg"
        base_name = os.path.splitext(image_name)[0]

        json_path = os.path.join(output_path, f"{base_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

        if save_detection or save_recognition or save_translation:
            try:
                image = load_image(image_path)
                if image is None:
                    self.logger.warning(f"Could not load image for visualization: {image_path}")
                    return
                
                boxes = [np.array(block["box"]) for block in results["blocks"]]
                classes = [block["class_name"] for block in results["blocks"]]
                confidences = [block["confidence"] for block in results["blocks"]]
                ids = [block["id"] for block in results["blocks"]]

                if save_detection:
                    labels = [
                        f"{i+1}: {classes[i]} ({confidences[i]:.2f})"
                        for i in range(len(boxes))
                    ]
                    
                    detection_viz = draw_boxes_with_labels(image.copy(), boxes, labels, classes)
                    detection_viz_path = os.path.join(output_path, f"{base_name}_detection.jpg")
                    save_image(detection_viz, detection_viz_path)
                    self.logger.info(f"Saved detection visualization to {detection_viz_path}")

                if save_recognition:
                    text_labels = [
                        f"{i+1}: {block.get('text', '')}"
                        for i, block in enumerate(results["blocks"])
                    ]
                    
                    text_viz = draw_boxes_with_labels(image.copy(), boxes, text_labels)
                    text_viz_path = os.path.join(output_path, f"{base_name}_text.jpg")
                    save_image(text_viz, text_viz_path)
                    self.logger.info(f"Saved text visualization to {text_viz_path}")

                if save_translation and any(block.get("translated_text") for block in results["blocks"]):
                    translation_labels = [
                        f"{i+1}: {block.get('translated_text', '')}"
                        for i, block in enumerate(results["blocks"])
                    ]
                    
                    translation_viz = draw_boxes_with_labels(image.copy(), boxes, translation_labels)
                    translation_viz_path = os.path.join(output_path, f"{base_name}_translation.jpg")
                    save_image(translation_viz, translation_viz_path)
                    self.logger.info(f"Saved translation visualization to {translation_viz_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create visualizations: {e}")
    
    async def process_chapter(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        save_segments: bool = False,
        save_detection: bool = False,
        save_recognition: bool = False,
        save_translation: bool = False,
        save_crops: bool = False
    ) -> Dict[str, Any]:
        """
        Process all images in a chapter directory using batch processing.
        
        This method processes the chapter in a "widthwise" manner: first segmentation 
        for all images, then detection for all images, then recognition, OCR correction,
        and finally translation. This approach is more efficient for API usage.
        
        Args:
            input_dir: Directory containing chapter images
            output_dir: Directory to save output files (None for no saving)
            save_segments: Whether to save segmented images
            save_detection: Whether to save detection visualizations
            save_recognition: Whether to save recognition visualizations
            save_translation: Whether to save translation visualizations
            save_crops: Whether to save cropped text blocks
            
        Returns:
            Dictionary containing all chapter processing results
        """
        self.logger.info(f"Processing chapter: {input_dir}")
        start_time = time.time()

        if not os.path.isdir(input_dir):
            self.logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create task-based directory structure
            if save_segments:
                os.makedirs(os.path.join(output_dir, "segments"), exist_ok=True)
            if save_detection:
                os.makedirs(os.path.join(output_dir, "detection"), exist_ok=True)
            if save_recognition:
                os.makedirs(os.path.join(output_dir, "recognition"), exist_ok=True)
            if save_translation:
                os.makedirs(os.path.join(output_dir, "translation"), exist_ok=True)
            if save_crops:
                os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)

        # STEP 1: SEGMENTATION (if enabled)
        self.logger.info("STEP 1: Segmentation")
        seg_time = time.time()
        
        image_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(input_dir, file))
        
        image_files = natsorted(image_files)
        
        if not image_files:
            self.logger.error(f"No image files found in: {input_dir}")
            raise ValueError(f"No image files found in: {input_dir}")
                
        self.logger.info(f"Found {len(image_files)} images to process")
        
        segments = []
        segment_paths = []
        segment_sources = []
        
        if self.use_segmentation:
            self.logger.info("Segmenting long comics")
            try:
                segments_dir = None
                if output_dir and save_segments:
                    segments_dir = os.path.join(output_dir, "segments")

                segment_results = segment_by_seam_carving(
                    input_path=input_dir,
                    min_height=self.min_seg_height,
                    save_crops=save_segments,
                    output_path=segments_dir,
                    resize=True
                )

                segments = segment_results['segments']
                
                if save_segments and segments_dir:
                    for i in range(len(segments)):
                        segment_filename = f"segment_{i+1:03d}.png"
                        segment_path = os.path.join(segments_dir, segment_filename)
                        segment_paths.append(segment_path)

                        # Might need a better mapping
                        segment_sources.append(image_files[min(i, len(image_files)-1)])
                else:
                    segment_paths = [None] * len(segments)
                    # Approximate mapping of sources
                    for i in range(len(segments)):
                        segment_sources.append(image_files[min(i, len(image_files)-1)])
                
                self.logger.info(f"Created {len(segments)} segments from {len(image_files)} images")
                    
            except Exception as e:
                self.logger.warning(f"Segmentation failed: {e}, falling back to processing original images")
                self.logger.exception("Segmentation error details:")
                self.use_segmentation = False
                segments = []
                segment_paths = []
                segment_sources = []
        
        if not self.use_segmentation or not segments:
            self.logger.info("Using original images (no segmentation)")
            segments = [] 
            segment_paths = image_files
            segment_sources = image_files
        
        # STEP 2: DETECTION & SPATIAL PROCESSING
        self.logger.info("STEP 2: Detection and spatial processing")
        detect_time = time.time()
        
        all_blocks = []
        all_frames = []
        frame_to_image_map = {}
        
        for i, image_path in enumerate(segment_paths):
            try:
                self.logger.info(f"Detecting text blocks in frame {i+1}/{len(segment_paths)}")

                if self.use_segmentation and segments and not image_path:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_path = temp_file.name
                        cv2.imwrite(temp_path, segments[i])
                        image_path = temp_path
                        temp_created = True
                else:
                    temp_created = False
                
                # Detect text blocks
                detected_blocks = self.detector.process_image(image_path, conf=0.5, iou=0.7)
                
                # Apply spatial processing
                text_blocks = [
                    TextBlock(
                        box=np.array(block["box"]),
                        class_name=block["class_name"],
                        confidence=block["confidence"],
                        id=block["id"]
                    )
                    for block in detected_blocks
                ]
                
                sorted_blocks, separator_lines = process_text_blocks(text_blocks)
                
                processed_blocks = []
                for j, (block, separators) in enumerate(zip(sorted_blocks, separator_lines)):
                    block_dict = {
                        "id": len(all_blocks) + j,  # Global ID across all frames
                        "frame_id": len(all_frames),  # Which frame this block belongs to
                        "box": block.box.tolist(),
                        "class_name": block.class_name,
                        "confidence": block.confidence,
                        "separators": [sep.tolist() for sep in separators] if separators else None,
                        "source_image": segment_sources[i] if i < len(segment_sources) else None
                    }
                    processed_blocks.append(block_dict)

                if output_dir and save_detection:
                    try:
                        detection_dir = os.path.join(output_dir, "detection")
                        
                        image = load_image(image_path)
                        if image is None:
                            self.logger.warning(f"Could not load image for visualization: {image_path}")
                        else:
                            boxes = [np.array(block["box"]) for block in processed_blocks]
                            classes = [block["class_name"] for block in processed_blocks]
                            confidences = [block["confidence"] for block in processed_blocks]
                            
                            labels = [
                                f"{block['id']}: {block['class_name']} ({block['confidence']:.2f})"
                                for block in processed_blocks
                            ]
                            
                            detection_viz = draw_boxes_with_labels(image.copy(), boxes, labels, classes)
                            
                            # Add separator lines to the visualization
                            for j, block_dict in enumerate(processed_blocks):
                                if block_dict.get("separators"):
                                    block = TextBlock(
                                        box=np.array(block_dict["box"]), 
                                        class_name=block_dict["class_name"],
                                        confidence=block_dict["confidence"], 
                                        id=block_dict["id"]
                                    )
                                    separators = [np.array(sep) for sep in block_dict["separators"]]
                                    detection_viz = visualize_separator_lines(detection_viz, [block], [separators])

                            detection_viz_path = os.path.join(detection_dir, f"frame_{i+1:03d}.jpg")
                            save_image(detection_viz, detection_viz_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to save detection visualization: {e}")

                if temp_created:
                    try:
                        os.unlink(image_path)
                    except:
                        pass

                all_blocks.extend(processed_blocks)
                
                frame_info = {
                    "id": len(all_frames),
                    "image_path": image_path,
                    "source_image": segment_sources[i] if i < len(segment_sources) else None,
                    "blocks": processed_blocks,
                    "num_blocks": len(processed_blocks)
                }
                all_frames.append(frame_info)
                frame_to_image_map[len(all_frames) - 1] = segment_sources[i] if i < len(segment_sources) else None
                
            except Exception as e:
                self.logger.error(f"Error in detection step for frame {i+1}: {e}")
                # Continue with next frame
        
        if not all_frames:
            self.logger.error("No frames were successfully processed in the detection step")
            raise RuntimeError("Detection failed for all images")
                
        self.logger.info(f"Detection completed: {len(all_blocks)} blocks detected across {len(all_frames)} frames")
        
        # STEP 3: RECOGNITION
        self.logger.info("STEP 3: Text recognition")
        ocr_time = time.time()
        
        for i, frame in enumerate(all_frames):
            try:
                image_path = frame["image_path"]
                self.logger.info(f"Recognizing text in frame {i+1}/{len(all_frames)}")

                frame_blocks = frame["blocks"]

                if not frame_blocks:
                    continue
                
                # Process blocks with OCR
                ocr_results = self.recognizer.process_image_with_blocks(
                    image_path, frame_blocks, apply_separators=self.apply_separators
                )

                if output_dir and save_crops:
                    try:
                        image = load_image(image_path)
                        if image is not None:
                            crops_dir = os.path.join(output_dir, "crops")
                            
                            for block in frame_blocks:
                                try:
                                    box = np.array(block["box"]).astype(int)
                                    x1, y1, x2, y2 = box
                                    crop = image[y1:y2, x1:x2].copy()
                                    
                                    # Apply separator masks if needed
                                    if self.apply_separators and block.get("separators"):
                                        local_separators = []
                                        for sep in block["separators"]:
                                            local_p1 = np.array([sep[0][0] - x1, sep[0][1] - y1], dtype=np.int32)
                                            local_p2 = np.array([sep[1][0] - x1, sep[1][1] - y1], dtype=np.int32)
                                            local_sep = np.array([local_p1, local_p2])
                                            local_separators.append(local_sep)
                                        
                                        from comicglot.utils.spatial_processor import apply_separator_masks
                                        crop = apply_separator_masks(crop, local_separators)

                                    crop_path = os.path.join(crops_dir, f"frame_{i+1:03d}_block_{block['id']:03d}_{block['class_name']}.jpg")
                                    save_image(crop, crop_path)
                                except Exception as e:
                                    self.logger.warning(f"Failed to save crop for block {block['id']}: {e}")
                        else:
                            self.logger.warning(f"Could not load image for crops: {image_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save crops: {e}")
                
                # If dual OCR is enabled, perform secondary recognition
                if self.dual_ocr and self.secondary_recognizer:
                    self.logger.info(f"Running secondary OCR for frame {i+1}")
                    secondary_ocr_results = self.secondary_recognizer.process_image_with_blocks(
                        image_path, frame_blocks, apply_separators=self.apply_separators
                    )

                    for j, (primary_result, secondary_result) in enumerate(zip(ocr_results, secondary_ocr_results)):
                        ocr_results[j]["secondary_text"] = secondary_result["text"]

                for j, ocr_result in enumerate(ocr_results):
                    block_id = frame_blocks[j]["id"]

                    all_block_idx = next((idx for idx, b in enumerate(all_blocks) if b["id"] == block_id), None)
                    
                    if all_block_idx is not None:
                        all_blocks[all_block_idx].update(ocr_result)

                    frame_blocks[j].update(ocr_result)

                if output_dir and save_recognition and not (self.use_correction and self.dual_ocr):
                    try:
                        recognition_dir = os.path.join(output_dir, "recognition")
                        os.makedirs(recognition_dir, exist_ok=True)

                        detection_path = None
                        if save_detection:
                            detection_dir = os.path.join(output_dir, "detection")
                            detection_path = os.path.join(detection_dir, f"frame_{i+1:03d}.jpg")

                        if detection_path and os.path.exists(detection_path):
                            base_image = load_image(detection_path)
                        else:
                            base_image = load_image(image_path)
                            
                            if base_image is not None:
                                text_blocks = []
                                separator_lines_lists = []
                                
                                for block in frame_blocks:
                                    text_block = TextBlock(
                                        box=np.array(block["box"]),
                                        class_name=block["class_name"],
                                        confidence=block["confidence"],
                                        id=block["id"]
                                    )
                                    text_blocks.append(text_block)
                                    
                                    if block.get("separators"):
                                        separator_lines = [np.array(sep) for sep in block["separators"]]
                                        separator_lines_lists.append(separator_lines)
                                    else:
                                        separator_lines_lists.append([])

                                bubble_boxes = [block.box for block in text_blocks]
                                classes = [block.class_name for block in text_blocks]
                                labels = [f"{block.class_name} ({block.confidence:.2f})" for block in text_blocks]
                                
                                base_image = draw_boxes_with_labels(base_image, bubble_boxes, labels, classes)
                                base_image = visualize_separator_lines(base_image, text_blocks, separator_lines_lists)
                        
                        if base_image is not None:
                            # Draw CRAFT detected lines with their text
                            line_boxes = []
                            line_texts = []
                            
                            for block in frame_blocks:

                                x_min, y_min, x_max, y_max = map(int, block["box"])
                                
                                if "text_lines" in block and block["text_lines"]:
                                    for line in block["text_lines"]:
                                        if "box" in line and "text" in line:
                                            line_boxes.append(np.array(line["box"]))
                                            line_texts.append(line["text"])

                            if line_boxes:
                                final_viz = draw_boxes_with_labels(
                                    base_image,                             # Use detection visualization as base
                                    line_boxes,                             # CRAFT detected line boxes in global coordinates
                                    line_texts,                             # Recognized text as labels
                                )
                            else:
                                final_viz = base_image

                            recognition_path = os.path.join(recognition_dir, f"frame_{i+1:03d}.jpg")
                            save_image(final_viz, recognition_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to save recognition visualization: {e}")

            except Exception as e:
                self.logger.error(f"Error in recognition step for frame {i+1}: {e}")

        self.logger.info(f"Recognition completed for {len(all_frames)} frames")
        
        # STEP 4: OCR CORRECTION (if enabled)
        correction_time = time.time()
        if self.use_correction and self.corrector and self.dual_ocr:
            self.logger.info("STEP 4: OCR correction")
            
            batch_correction_inputs = []
            blocks_needing_correction = []
            correction_contexts = []
            
            for block in all_blocks:
                primary_text = block.get("text", "").strip()
                secondary_text = block.get("secondary_text", "").strip()
                
                # Skip correction if texts are identical or too similar
                if self._are_texts_equivalent(primary_text, secondary_text):
                    block["corrected_text"] = primary_text
                    self.logger.debug(f"Block {block['id']}: Texts are equivalent, skipping correction")
                elif not primary_text or not secondary_text:
                    # Skip if either OCR result is empty
                    block["corrected_text"] = primary_text or secondary_text
                    self.logger.debug(f"Block {block['id']}: One OCR result is empty, using non-empty result")
                else:
                    # Both texts have content and they're different - needs correction
                    batch_correction_inputs.append([primary_text, secondary_text])
                    blocks_needing_correction.append(block)
                    if self.correction_context:
                        idx = all_blocks.index(block)
                        prev = [b.get("corrected_text") or b.get("text", "") for b in all_blocks[max(0, idx - 3):idx]]
                        prev = [t for t in prev if t]
                        correction_contexts.append(
                            f"Text block type: {block['class_name']}." +
                            (" Previous lines on the page: " + " / ".join(prev) if prev else ""))
                    else:
                        correction_contexts.append(None)
                    self.logger.debug(f"Block {block['id']}: Texts differ, adding to correction batch")
            
            if batch_correction_inputs:
                self.logger.info(f"Sending {len(batch_correction_inputs)} blocks for OCR correction")

                if len(batch_correction_inputs) > 0:
                    sample_count = min(3, len(batch_correction_inputs))
                    for i in range(sample_count):
                        self.logger.debug(f"Correction sample {i+1}:")
                        self.logger.debug(f"  OCR1: {batch_correction_inputs[i][0]}")
                        self.logger.debug(f"  OCR2: {batch_correction_inputs[i][1]}")

                corrected_texts = await self.corrector.batch_correct_texts(
                    batch_correction_inputs, contexts=correction_contexts, language=self.source_lang
                )

                for i, block in enumerate(blocks_needing_correction):
                    if i < len(corrected_texts):
                        block["corrected_text"] = corrected_texts[i]
                        self.logger.debug(f"Block {block['id']}: Corrected text: {corrected_texts[i]}")
                    else:
                        block["corrected_text"] = block.get("text", "")
                        self.logger.warning(f"Block {block['id']}: No correction received, using primary OCR")
            else:
                self.logger.info("No texts need correction")

            if output_dir and save_recognition:
                self.logger.info("Creating recognition visualizations with corrected text")
                
                recognition_dir = os.path.join(output_dir, "recognition")
                os.makedirs(recognition_dir, exist_ok=True)
                
                for i, frame in enumerate(all_frames):
                    try:
                        image_path = frame["image_path"]
                        frame_blocks = frame["blocks"]
                        
                        if not frame_blocks:
                            continue
                        
                        detection_path = None
                        if save_detection:
                            detection_dir = os.path.join(output_dir, "detection")
                            detection_path = os.path.join(detection_dir, f"frame_{i+1:03d}.jpg")
                        
                        if detection_path and os.path.exists(detection_path):
                            base_image = load_image(detection_path)
                        else:
                            base_image = load_image(image_path)
                        
                        if base_image is not None:
                            text_blocks = []
                            separator_lines_lists = []
                            
                            for block in frame_blocks:
                                text_block = TextBlock(
                                    box=np.array(block["box"]),
                                    class_name=block["class_name"],
                                    confidence=block["confidence"],
                                    id=block["id"]
                                )
                                text_blocks.append(text_block)
                                
                                if block.get("separators"):
                                    separator_lines = [np.array(sep) for sep in block["separators"]]
                                    separator_lines_lists.append(separator_lines)
                                else:
                                    separator_lines_lists.append([])
                            
                            bubble_boxes = [block.box for block in text_blocks]
                            classes = [block.class_name for block in text_blocks]
                            labels = [f"{block.class_name} ({block.confidence:.2f})" for block in text_blocks]
                            
                            base_image = draw_boxes_with_labels(base_image, bubble_boxes, labels, classes)
                            base_image = visualize_separator_lines(base_image, text_blocks, separator_lines_lists)
                        
                            line_boxes = []
                            line_texts = []
                            
                            for block in frame_blocks:
                                x_min, y_min, x_max, y_max = map(int, block["box"])
                                
                                if "text_lines" in block and block["text_lines"]:
                                    for line_idx, line in enumerate(block["text_lines"]):
                                        if "box" in line and "text" in line:
                                            line_box = np.array(line["box"])
                                            
                                            if len(block["text_lines"]) == 1 and "corrected_text" in block:
                                                line_text = block["corrected_text"]
                                            else:
                                                line_text = line["text"]
                                                
                                            line_boxes.append(line_box)
                                            line_texts.append(line_text)
                            
                            if line_boxes:
                                final_viz = draw_boxes_with_labels(
                                    base_image,
                                    line_boxes,
                                    line_texts
                                )
                            else:
                                final_viz = base_image
                            
                            recognition_path = os.path.join(recognition_dir, f"frame_{i+1:03d}.jpg")
                            save_image(final_viz, recognition_path)
                            self.logger.debug(f"Saved recognition visualization for frame {i+1} with corrected text")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to save corrected recognition visualization for frame {i+1}: {e}")
                    
        elif not self.dual_ocr and self.use_correction:
            self.logger.warning("OCR correction enabled but dual OCR is disabled. Skipping correction.")
            for block in all_blocks:
                block["corrected_text"] = block.get("text", "")
        else:
            # Just use the primary OCR results
            for block in all_blocks:
                block["corrected_text"] = block.get("text", "")
        
        # STEP 5: TRANSLATION (if enabled)
        translate_time = time.time()
        if self.translator:
            self.logger.info("STEP 5: Translation")

            batch_texts = []
            batch_bubble_types = []
            blocks_to_translate = []
            
            for block in all_blocks:
                text = block.get("corrected_text", "") or block.get("text", "")
                if not text.strip():
                    block["translated_text"] = ""
                    continue
                
                batch_texts.append(text)
                batch_bubble_types.append(block["class_name"])
                blocks_to_translate.append(block)

            if batch_texts:
                self.logger.info(f"Translating {len(batch_texts)} blocks using built-in context generation")

                guide_types = [self.translator.guide_class(t, self.class_condition) for t in batch_bubble_types]
                translated_texts = await self.translator.batch_translate(
                    texts=batch_texts,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    bubble_types=guide_types,
                    context_strategy="prev" if self.context_window > 0 else None,
                    context_window_size=max(self.context_window, 1)
                )
                for block, g in zip(blocks_to_translate, guide_types):
                    block["translation_guide_class"] = g

                for i, block in enumerate(blocks_to_translate):
                    if i < len(translated_texts):
                        block["translated_text"] = translated_texts[i]
                    else:
                        block["translated_text"] = ""
            else:
                self.logger.info("No blocks to translate")

            if output_dir and save_translation:
                translation_dir = os.path.join(output_dir, "translation")
                
                for i, frame in enumerate(all_frames):
                    try:
                        image_path = frame["image_path"]
                        image = load_image(image_path)
                        
                        if image is not None:
                            boxes = [np.array(block["box"]) for block in frame["blocks"]]
                            
                            translation_labels = [
                                f"{block['id']}: {block.get('translated_text', '')}"
                                for block in frame["blocks"]
                            ]
                            
                            translation_viz = draw_boxes_with_labels(image.copy(), boxes, translation_labels)
                            translation_viz_path = os.path.join(translation_dir, f"frame_{i+1:03d}.jpg")
                            save_image(translation_viz, translation_viz_path)
                        else:
                            self.logger.warning(f"Could not load image for visualization: {image_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save translation visualization for frame {i+1}: {e}")
        else:
            for block in all_blocks:
                block["translated_text"] = ""

        end_time = time.time()

        timing = {
            "segmentation": detect_time - seg_time,
            "detection": ocr_time - detect_time,
            "recognition": correction_time - ocr_time,
            "correction": translate_time - correction_time if self.use_correction else 0,
            "translation": end_time - translate_time if self.translator else 0,
            "total": end_time - start_time
        }
        
        results = {
            "chapter_path": input_dir,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_frames": len(all_frames),
            "num_blocks": len(all_blocks),
            "timing": timing,
            "frames": all_frames,
            "config": {
                "detection_model": self.detection_model_type,
                "recognition_model": self.recognition_model_type,
                "translator": self.translator_type,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "apply_separators": self.apply_separators,
                "dual_ocr": self.dual_ocr,
                "use_correction": self.use_correction,
                "use_segmentation": self.use_segmentation,
                "correction_context": self.correction_context,
                "class_condition": self.class_condition,
                "context_window": self.context_window,
                "arbitrate_punctuation": self.arbitrate_punctuation,
            },
            "provenance": self._provenance()
        }

        if output_dir:
            self._save_chapter_outputs(results, output_dir)
        
        return results

    def _save_chapter_outputs(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save chapter processing results to disk in a structured format.
        
        Args:
            results: Processing results dictionary
            output_dir: Base output directory
        """
        self.logger.info(f"Saving chapter processing results to {output_dir}")
        import json

        summary_path = os.path.join(output_dir, "chapter_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary = {
                "chapter_path": results["chapter_path"],
                "processed_time": results["processed_time"],
                "num_frames": results["num_frames"],
                "num_blocks": results["num_blocks"],
                "timing": results["timing"],
                "config": results["config"]
            }
            json.dump(summary, f, ensure_ascii=False, indent=2)

        full_results_path = os.path.join(output_dir, "chapter_full_results.json")
        with open(full_results_path, 'w', encoding='utf-8') as f:
            import json
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
        
        text_path = os.path.join(output_dir, "recognized_text.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            for frame in results["frames"]:
                frame_id = frame["id"]
                source_image = frame.get("source_image", "unknown_source")
                if isinstance(source_image, str):
                    source_name = os.path.basename(source_image)
                else:
                    source_name = f"frame_{frame_id}"
                    
                f.write(f"Frame {frame_id} (Source: {source_name}):\n")
                f.write("-" * 50 + "\n")
                
                for block in frame["blocks"]:
                    block_id = block["id"]
                    class_name = block["class_name"]
                    text = block.get("corrected_text", "") or block.get("text", "")
                    
                    if text.strip():
                        f.write(f"Block {block_id} ({class_name}):\n")
                        f.write(f"{text}\n\n")
                
                f.write("\n\n")

        if any(block.get("translated_text") for frame in results["frames"] for block in frame["blocks"]):
            translation_path = os.path.join(output_dir, "translated_text.txt")
            with open(translation_path, 'w', encoding='utf-8') as f:
                for frame in results["frames"]:
                    frame_id = frame["id"]
                    source_image = frame.get("source_image", "unknown_source")
                    if isinstance(source_image, str):
                        source_name = os.path.basename(source_image)
                    else:
                        source_name = f"frame_{frame_id}"
                        
                    f.write(f"Frame {frame_id} (Source: {source_name}):\n")
                    f.write("-" * 50 + "\n")
                    
                    for block in frame["blocks"]:
                        block_id = block["id"]
                        class_name = block["class_name"]
                        translated_text = block.get("translated_text", "")
                        
                        if translated_text.strip():
                            f.write(f"Block {block_id} ({class_name}):\n")
                            f.write(f"{translated_text}\n\n")
                    
                    f.write("\n\n")
        
        self.logger.info(f"Saved chapter processing results to {output_dir}")