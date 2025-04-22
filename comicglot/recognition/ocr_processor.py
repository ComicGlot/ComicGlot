"""
OCR processor module for ComicGlot.

This module provides functionality to detect and recognize text in images
using a combination of text detection and recognition models.
"""

import os
import logging
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class OCRProcessor:
    """
    OCR Processor for detecting and recognizing text in comic images.
    
    This class integrates multiple text detection and recognition models to process
    text in comic images, particularly focusing on Asian languages.
    """
    
    def __init__(
        self, 
        language: List[str] = ['ko', 'en'],
        detection_model: str = 'craft', 
        recognition_model: str = 'trocr',
        custom_model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the OCR processor.
        
        Args:
            language: List of language codes for OCR (e.g., ['ko', 'en'])
            detection_model: Detection model to use ('craft' or 'east')
            recognition_model: Recognition model to use ('trocr' or 'easyocr')
            custom_model_path: Path to custom model weights (optional)
            use_gpu: Whether to use GPU for processing
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.language = language
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize text detection
        self._init_text_detector()
        
        # Initialize text recognition
        self._init_text_recognizer(custom_model_path)
        
        self.logger.info("OCR processor initialized successfully")
    
    def _init_text_detector(self):
        """Initialize the text detection model."""
        if self.detection_model == 'craft':
            try:
                # Import EasyOCR for CRAFT
                from easyocr import Reader
                self.logger.info(f"Initializing CRAFT detector via EasyOCR for languages: {self.language}")
                self.text_detector = Reader(self.language, gpu=(self.device.type == 'cuda'), 
                                           detector=True, recognizer=False)
            except ImportError:
                self.logger.error("Failed to import EasyOCR. Please install with: pip install easyocr")
                raise
        elif self.detection_model == 'east':
            self.logger.error("EAST detector not yet implemented")
            raise NotImplementedError("EAST detector not yet implemented")
        else:
            self.logger.error(f"Unknown detection model: {self.detection_model}")
            raise ValueError(f"Unknown detection model: {self.detection_model}")
    
    def _init_text_recognizer(self, custom_model_path: Optional[str] = None):
        """
        Initialize the text recognition model.
        
        Args:
            custom_model_path: Path to custom model weights (optional)
        """
        if self.recognition_model == 'trocr':
            try:
                from transformers import (
                    TrOCRProcessor, 
                    AutoTokenizer,
                    VisionEncoderDecoderModel,
                )
                
                self.logger.info("Initializing TrOCR recognizer")
                
                # Initialize TrOCR
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                
                if custom_model_path:
                    self.logger.info(f"Loading custom TrOCR model from {custom_model_path}")
                    self.trocr_tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
                    self.trocr_model = VisionEncoderDecoderModel.from_pretrained(custom_model_path).to(self.device)
                else:
                    # Use pretrained model based on language
                    if 'ko' in self.language:
                        model_name = "team-lucid/trocr-small-korean"
                    else:
                        model_name = "microsoft/trocr-base-handwritten"
                    
                    self.logger.info(f"Loading pretrained TrOCR model: {model_name}")
                    self.trocr_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
                
            except ImportError:
                self.logger.error("Failed to import transformers. Please install with: pip install transformers")
                raise
        
        elif self.recognition_model == 'easyocr':
            # EasyOCR already initialized in detector if we're using it
            if self.detection_model != 'easyocr':
                try:
                    from easyocr import Reader
                    self.logger.info(f"Initializing EasyOCR recognizer for languages: {self.language}")
                    self.text_recognizer = Reader(self.language, gpu=(self.device.type == 'cuda'))
                except ImportError:
                    self.logger.error("Failed to import EasyOCR. Please install with: pip install easyocr")
                    raise
        else:
            self.logger.error(f"Unknown recognition model: {self.recognition_model}")
            raise ValueError(f"Unknown recognition model: {self.recognition_model}")
    
    def detect_text_lines(
        self, 
        image: np.ndarray, 
        min_size: int = 20,
        paragraph_mode: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines in an image.
        
        Args:
            image: Input image as numpy array
            min_size: Minimum size of text boxes to be considered
            paragraph_mode: Whether to group lines into paragraphs
            
        Returns:
            List of bounding boxes in format (x_min, y_min, x_max, y_max)
        """
        self.logger.info("Detecting text lines")
        
        if self.detection_model == 'craft':
            horizontal_list, _ = self.text_detector.detect(image, min_size=min_size)
            # Convert from (x_min, x_max, y_min, y_max) to (x_min, y_min, x_max, y_max)
            boxes = [(x_min, y_min, x_max, y_max) for (x_min, x_max, y_min, y_max) in horizontal_list[0]]
            
            if paragraph_mode and len(boxes) > 1:
                # Group text lines into paragraphs using EasyOCR's grouping logic
                # This is a simplified version - full implementation would use get_paragraph
                from comicglot.utils.text_utils import group_text_lines
                boxes = group_text_lines(boxes)
            
            return boxes
        
        elif self.detection_model == 'east':
            # Placeholder for EAST implementation
            raise NotImplementedError("EAST detector not yet implemented")
        
        else:
            self.logger.error(f"Unknown detection model: {self.detection_model}")
            raise ValueError(f"Unknown detection model: {self.detection_model}")
    
    def recognize_text(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[str]:
        """
        Recognize text in specified regions of an image.
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes in format (x_min, y_min, x_max, y_max)
            
        Returns:
            List of recognized text strings
        """
        self.logger.info("Recognizing text in detected regions")
        
        if self.recognition_model == 'trocr':
            return self._perform_trocr_recognition(image, boxes)
        
        elif self.recognition_model == 'easyocr':
            return self._perform_easyocr_recognition(image, boxes)
        
        else:
            self.logger.error(f"Unknown recognition model: {self.recognition_model}")
            raise ValueError(f"Unknown recognition model: {self.recognition_model}")
    
    def _perform_trocr_recognition(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[str]:
        """
        Perform text recognition using TrOCR.
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes to recognize
            
        Returns:
            List of recognized text strings
        """
        self.logger.info("Recognizing text using TrOCR")
        results = []
        
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Crop the region
            try:
                cropped = image[y_min:y_max, x_min:x_max]
                
                # Convert to RGB for TrOCR
                if len(cropped.shape) == 2:  # Grayscale
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
                else:  # BGR
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                # Process with TrOCR
                pixel_values = self.trocr_processor(cropped_rgb, return_tensors="pt").pixel_values.to(self.device)
                
                # Configure generation
                gen_config = None
                if hasattr(self.trocr_model, 'generation_config'):
                    gen_config = self.trocr_model.generation_config
                else:
                    gen_config = {
                        "max_length": 128,
                        "early_stopping": True,
                        "num_beams": 4,
                        "no_repeat_ngram_size": 3,
                    }
                
                # Generate text
                if isinstance(gen_config, dict):
                    generated_ids = self.trocr_model.generate(pixel_values, **gen_config)
                else:
                    generated_ids = self.trocr_model.generate(pixel_values, generation_config=gen_config)
                
                # Decode text
                generated_text = self.trocr_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                results.append("")
                
        return results
    
    def _perform_easyocr_recognition(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[str]:
        """
        Perform text recognition using EasyOCR.
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes to recognize
            
        Returns:
            List of recognized text strings
        """
        self.logger.info("Recognizing text using EasyOCR")
        results = []
        
        # Use text_detector or text_recognizer based on what's available
        recognizer = getattr(self, 'text_recognizer', getattr(self, 'text_detector', None))
        
        if recognizer is None:
            self.logger.error("EasyOCR recognizer not initialized")
            raise RuntimeError("EasyOCR recognizer not initialized")
        
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Crop the region
            try:
                cropped = image[y_min:y_max, x_min:x_max]
                
                # Recognize with EasyOCR
                text_result = recognizer.recognize(cropped, detail=0, paragraph=False)
                
                # Join multiple text entries
                text = " ".join(text_result).strip()
                results.append(text)
                
            except Exception as e:
                self.logger.warning(f"Error processing region {box}: {str(e)}")
                results.append("")
        
        return results
    
    def perform_detection_and_recognition(
        self,
        image: Union[str, np.ndarray],
        use_trocr: bool = True,
        detection: bool = True,
        min_block_size: int = 20,
        paragraph_mode: bool = False
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Detect and recognize text regions in an image.
        
        Args:
            image: Input image as numpy array or path to image file
            use_trocr: Whether to use TrOCR for recognition (fallback to EasyOCR if False)
            detection: Whether to perform text line detection (if False, treats the whole image as one text block)
            min_block_size: Minimum size of text boxes to be considered
            paragraph_mode: Whether to group lines into paragraphs
            
        Returns:
            List of tuples, each containing (bounding_box, recognized_text)
        """
        self.logger.info("Starting text detection and recognition process")
        
        # Load image if path is provided
        if isinstance(image, str):
            self.logger.info(f"Loading image from {image}")
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        
        # Detect text regions or use whole image
        if detection:
            boxes = self.detect_text_lines(image, min_size=min_block_size, paragraph_mode=paragraph_mode)
        else:
            height, width = image.shape[:2]
            boxes = [(0, 0, width, height)]
        
        # Set recognition model based on parameter
        original_recognition_model = self.recognition_model
        if use_trocr:
            self.recognition_model = 'trocr'
        else:
            self.recognition_model = 'easyocr'
        
        # Perform recognition
        try:
            texts = self.recognize_text(image, boxes)
        except Exception as e:
            self.logger.warning(f"Recognition failed with {self.recognition_model}: {e}. Trying fallback.")
            # Fallback to the other method
            self.recognition_model = 'easyocr' if self.recognition_model == 'trocr' else 'trocr'
            try:
                texts = self.recognize_text(image, boxes)
            except Exception as e2:
                self.logger.error(f"Fallback recognition also failed: {e2}")
                texts = [""] * len(boxes)
        finally:
            # Restore original setting
            self.recognition_model = original_recognition_model
        
        # Combine boxes and texts
        return list(zip(boxes, texts))
    
    def process_text_block(
        self,
        image: Union[str, np.ndarray],
        block_class: str,
        detect_lines: Optional[bool] = None,
        **kwargs
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Process a text block with class-specific settings.
        
        Args:
            image: Image containing the text block
            block_class: Class name of the text block (e.g., 'speech_bubble', 'sfx_sound')
            detect_lines: Whether to detect lines within the block (overrides class-specific default)
            **kwargs: Additional arguments to pass to perform_detection_and_recognition
            
        Returns:
            List of tuples, each containing (bounding_box, recognized_text)
        """
        self.logger.info(f"Processing {block_class} text block")
        
        # Default processing settings based on block class
        detection_settings = {
            'speech_bubble': True,
            'cloud_bubble': True,
            'scream_bubble': True,
            'spiked_bubble': True,
            'system_bubble': True,
            'think_bubble': True,
            'standalone_text': False,
            'sfx_sound': False,
            'hexagon_bubble': True,
            'square_bubble': True,
            'wiggle_bubble': True,
            'title_panel': False,
            'background_text': False
        }
        
        # Determine if we should detect lines
        if detect_lines is None:
            # Use class-specific default
            detect_lines = detection_settings.get(block_class, True)
        
        # Additional class-specific settings
        recognition_settings = {
            'system_bubble': {'paragraph_mode': True},
            'sfx_sound': {'use_trocr': False},  # EasyOCR might be better for SFX
            'title_panel': {'min_block_size': 30},  # Larger min size for titles
        }
        
        # Combine settings
        process_kwargs = recognition_settings.get(block_class, {})
        process_kwargs.update(kwargs)
        
        # Process the block
        return self.perform_detection_and_recognition(
            image, 
            detection=detect_lines,
            **process_kwargs
        )
    
    def process_image_with_blocks(
        self,
        image_path: str,
        blocks: List[Dict[str, Any]],
        apply_separators: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process text blocks detected in an image.
        
        Args:
            image_path: Path to the image file
            blocks: List of detected blocks, each as a dictionary with keys:
                   'id', 'box', 'class_name', 'confidence', and optionally 'separator'
            apply_separators: Whether to apply separator masks to resolve overlapping blocks
            
        Returns:
            List of blocks with added 'text' and 'text_lines' fields
        """
        self.logger.info(f"Processing {len(blocks)} blocks in {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Process each block
        results = []
        for block in blocks:
            try:
                # Crop the block
                x_min, y_min, x_max, y_max = map(int, block['box'])
                cropped_block = image[y_min:y_max, x_min:x_max].copy()
                
                # Apply separator mask if needed
                if apply_separators and block.get('separator') is not None:
                    from comicglot.utils.spatial_processor import apply_separator_mask
                    separator_line = np.array(block['separator'])
                    
                    # Adjust separator line to local coordinates
                    separator_line[0][0] -= x_min
                    separator_line[0][1] -= y_min
                    separator_line[1][0] -= x_min
                    separator_line[1][1] -= y_min
                    
                    cropped_block = apply_separator_mask(cropped_block, separator_line)
                
                # Process the block based on its class
                ocr_results = self.process_text_block(
                    cropped_block,
                    block['class_name']
                )
                
                # Extract text
                block_text = " ".join([text for _, text in ocr_results]).strip()
                
                # Add OCR results to block
                block_result = block.copy()
                block_result['text'] = block_text
                block_result['text_lines'] = [
                    {'box': box, 'text': text}
                    for box, text in ocr_results
                ]
                
                results.append(block_result)
                
            except Exception as e:
                self.logger.error(f"Error processing block {block.get('id', '?')}: {e}")
                # Add error information and empty text
                block_result = block.copy()
                block_result['text'] = ""
                block_result['text_lines'] = []
                block_result['error'] = str(e)
                results.append(block_result)
        
        return resultsappend(generated_text)
                
            except Exception as e:
                self.logger.warning(f"Error processing region {box}: {str(e)}")
                results.