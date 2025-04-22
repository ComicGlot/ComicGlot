import os
from dotenv import load_dotenv

import torch
import unicodedata
import logging
import cv2
from typing import List, Tuple, Union
from easyocr import Reader
from transformers import (
    TrOCRProcessor, 
    AutoTokenizer, 
    # AutoModelForImageTextToText, # update to newer version to use. But problem with training compatibility
    VisionEncoderDecoderModel,
    GenerationConfig,
)

from utils import crop_image, load_image

class OCRProcessor:
    def __init__(self, language: List[str] = ['ko', 'en'], custom_model_path: str = None, use_gpu: bool = True):
        """
        Initializes EasyOCR and TrOCR for OCR processing.

        Args:
            language (str): Language code for OCR.
            use_gpu (bool): Whether to use GPU for EasyOCR.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logging.info(f"Using device: {self.device}")

        logging.info(f"Initializing EasyOCR for language: {language}")
        self.easyocr_reader = Reader(language, gpu=(self.device.type == 'cuda'))

        logging.info("Initializing TrOCR")

        if custom_model_path != None:
            logging.info(f"Starting local TrOCR model on {custom_model_path}")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(custom_model_path).to(self.device)
    
        else:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

            #self.trocr_tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")
            #self.trocr_model = VisionEncoderDecoderModel.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot").to(self.device)

            self.trocr_tokenizer = AutoTokenizer.from_pretrained("team-lucid/trocr-small-korean")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("team-lucid/trocr-small-korean").to(self.device)

            #self.trocr_tokenizer = AutoTokenizer.from_pretrained("E:/Projects/Python/Manhua_Scanner/trocr-finetuned")
            #self.trocr_model = VisionEncoderDecoderModel.from_pretrained("E:/Projects/Python/Manhua_Scanner/trocr-finetuned").to(self.device)
            
            #self.trocr_processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr") 
            #self.trocr_model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr").to(self.device)
            #self.trocr_tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

            #self.trocr_tokenizer = AutoTokenizer.from_pretrained("gg4ever/trOCR-final")
            #self.trocr_model = AutoModelForImageTextToText.from_pretrained("gg4ever/trOCR-final").to(self.device)
    
    def detect_craft_text_boxes(self, image: cv2.Mat, min_block_size: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using EasyOCR`s CRAFT.

        Args:
            image (cv2.Mat): Input image.
            min_block_size (int): Minimum size of text boxes to be considered.

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes for detected text regions.
        """
        logging.info("Detecting text boxes using CRAFT")
        horizontal_list, _ = self.easyocr_reader.detect(image, min_size=min_block_size)

        return horizontal_list[0]
    
    def detect_east_text_boxes(self, image: cv2.Mat, min_block_size: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using EAST algo.

        Args:
            image (cv2.Mat): Input image.
            min_block_size (int): Minimum size of text boxes to be considered.

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes for detected text regions.
        """
        pass

    def perform_easyocr(self, image: cv2.Mat, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Perform OCR using EasyOCR.

        Args:
            image (cv2.Mat): Preprocessed image.
            boxes (List[Tuple[int, int, int, int]]): Detected bounding boxes.

        Returns:
            List[str]: Recognized text for each bounding box.
        """
        logging.info("Recognizing text using EasyOCR")
        results = []
        for box in boxes:
            #x_min, y_min, x_max, y_max = box
            cropped = crop_image(image, (box))
            text = self.easyocr_reader.recognize(cropped, detail=0, paragraph=False)
            results.append(" ".join(text).strip())
        return results

    def perform_trocr(self, image: cv2.Mat, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Perform OCR using TrOCR.

        Args:
            image (cv2.Mat): Preprocessed image.
            boxes (List[Tuple[int, int, int, int]]): Detected bounding boxes.

        Returns:
            List[str]: Recognized text for each bounding box.
        """
        logging.info("Recognizing text using TrOCR")
        results = []

        gen_config = GenerationConfig(
            max_new_tokens=128,
            do_sample=True,
            early_stopping=True,
            temperature=0.5,
            num_beams=5,
        )
        for box in boxes:
            #x_min, y_min, x_max, y_max = box
            cropped = crop_image(image, (box))
            pil_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.trocr_model.generate(pixel_values, generation_config=gen_config) # was max_length=64, 
            #text = self.trocr_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            generated_text = self.trocr_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            #generated_text = unicodedata.normalize("NFC", generated_text) # For ddobokki/ko-trocr!!
            results.append(generated_text)
        return results

    def perform_detection_and_recognition(
        self, image: cv2.Mat, use_trocr: bool = True, detection: bool = True, use_craft: bool = True, 
        min_block_size: int = 20
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Detect and recognize text regions in an image.

        Args:
            image (cv2.Mat): Preprocessed image.
            use_trocr (bool): Whether to use TrOCR for recognition (fallback to EasyOCR if False).
            min_block_size (int): Minimum size of text boxes to be considered.

        Returns:
            List[Tuple[Tuple[int, int, int, int], str]]: List of (bounding box, recognized text) tuples.
        """
        logging.info("Starting text detection and recognition process")
        if detection:
            if use_craft:
                boxes = self.detect_craft_text_boxes(image, min_block_size=min_block_size)
            else:
                boxes = self.detect_east_text_boxes(image, min_block_size=min_block_size)
        else:
            height, width = image.shape[:2]
            boxes = [(0, width, 0, height)]
        
        transposed_boxes = [(x_min, y_min, x_max, y_max) for (x_min, x_max, y_min, y_max) in boxes]

        if use_trocr:
            try:
                texts = self.perform_trocr(image, transposed_boxes)
            except Exception as e:
                logging.warning(f"TrOCR failed with error: {e}. Falling back to EasyOCR.")
                texts = self.perform_easyocr(image, transposed_boxes)
        else:
            texts = self.perform_easyocr(image, transposed_boxes)
        
        return list(zip(transposed_boxes, texts))

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        #level=logging.DEBUG,
    )

    # Load environment variables from .env file
    load_dotenv()

    TROCR_MODEL_PATH = os.path.abspath(os.getenv("TROCR_MODEL_PATH", "./models/TrOCR/tuned"))
    INPUT_BUBBLE = os.path.abspath(os.getenv("INPUT_BUBBLE", "./data/bubbles/test.jpg"))
    INPUT_LINE = os.path.abspath(os.getenv("INPUT_LINE", "./data/lines/test.jpg"))
    LANGUAGES_STR = os.getenv("LANGUAGE", "ko,en")
    LANGUAGES = LANGUAGES_STR.split(',')

    # ocr_processor = OCRProcessor(language=LANGUAGES, use_gpu=True)
    ocr_processor = OCRProcessor(language=LANGUAGES, custom_model_path=TROCR_MODEL_PATH)

    image = load_image(INPUT_BUBBLE)
    # image = load_image(INPUT_LINE)

    results = ocr_processor.perform_detection_and_recognition(image, use_trocr=True, use_craft=True)
    # results = ocr_processor.perform_detection_and_recognition(image, use_trocr=True, use_craft=True, detection=False)
    for box, text in results:
        logging.info(f"Box: {box}, Text: {text}")
