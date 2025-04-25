"""
OCR Correction module for ComicGlot.

This module provides functionality to correct OCR errors by comparing 
outputs from different OCR models and using LLM to determine the most 
accurate version of the text.
"""

import os
import logging
import sys
from typing import Dict, List, Optional, Union, Any, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comicglot.utils.llm_utils import LLMProcessor

logger = logging.getLogger(__name__)

class OCRCorrector:
    """
    OCR Correction class that uses dual OCR results and LLM analysis
    to produce accurate text recognition.
    """
    
    def __init__(
        self,
        llm_processor: Optional[LLMProcessor] = None,
        model_name: str = "ocr_correction", # "ocr_correction"
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ):
        """
        Initialize the OCR corrector.
        
        Args:
            llm_processor: LLMProcessor instance (if None, creates a new one)
            model_name: Name of the LLM model to use (can use friendly names)
            system_prompt: System prompt to guide the LLM's behavior
            temperature: Temperature parameter for LLM generation
        """
        self.llm_processor = llm_processor or LLMProcessor(
            model_name=model_name,
            temperature=temperature
        )

        self.system_prompt = None

        # self.system_prompt = system_prompt or (
        #     "You are an expert OCR correction system. Your task is to analyze two different OCR outputs "
        #     "for the same text and determine the most accurate version, fixing any errors. "
        #     "Focus on logical consistency, correct grammar, and maintaining the original meaning. "
        #     "Be especially careful with names, numbers, and special terms. "
        #     "DO NOT add any information that is not present in either OCR version."
        # )
        
        self.temperature = temperature

        template_path = './data/prompts/correction_prompt.txt'
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            logger.info(f"OCR Corrector initialized with model: {self.llm_processor.model_name}")
        except Exception as e:
            logger.error(f"Failed to load OCR correction template: {str(e)}")
            raise
    
    def correct_text(
        self,
        ocr_results: List[str],
        context: Optional[str] = None,
        language: str = "korean",
        max_tokens: int = 2048
    ) -> str:
        """
        Correct OCR text by comparing multiple results.
        
        Args:
            ocr_results: List of OCR text results to compare
            context: Optional context about the text (e.g., bubble type, neighboring text)
            language: ISO language code of the text
            max_tokens: Maximum tokens for LLM response
            
        Returns:
            Corrected text string
        """
        if not ocr_results:
            logger.warning("No OCR results provided for correction")
            return ""
        
        if len(ocr_results) == 1:
            logger.info("Only one OCR result provided, skipping correction")
            return ocr_results[0]

        prompt = self.prompt_template.format(
            language=language,
            context_if_available=context or "No additional context available.",
            ocr_result_1=ocr_results[0],
            ocr_result_2=ocr_results[1]
        )
        
        try:
            logger.debug("Sending OCR correction prompt to LLM")
            response = self.llm_processor.process_with_claude_sync(
                prompt=prompt,
                # system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            print(response)
            
            # Extract the corrected text from the response
            if "CORRECTED_TEXT:" in response:
                corrected_text = response.split("CORRECTED_TEXT:", 1)[1].strip()
                logger.info("Successfully corrected OCR text")
                return corrected_text
            else:
                logger.warning("LLM response did not contain 'CORRECTED_TEXT:' marker")
                return ocr_results[0]  # Fallback to first OCR result
            
        except Exception as e:
            logger.error(f"Error in OCR correction: {str(e)}")
            return ocr_results[0]
    
    def batch_correct_texts(
        self,
        ocr_results_list: List[List[str]],
        contexts: Optional[List[str]] = None,
        language: str = "ko",
        max_tokens: int = 2048,
        concurrency_limit: int = 5
    ) -> List[str]:
        """
        Correct multiple OCR texts in batch.
        
        Args:
            ocr_results_list: List of OCR result lists to compare
            contexts: Optional contexts for each text
            language: ISO language code of the text
            max_tokens: Maximum tokens for LLM response
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            List of corrected text strings
        """
        if not ocr_results_list:
            return []
        
        if contexts is None:
            contexts = [None] * len(ocr_results_list)

        prompts = []
        valid_indices = []
        
        for i, ocr_results in enumerate(ocr_results_list):
            if not ocr_results:
                prompts.append("")  # Empty prompt for empty results
                continue
                
            if len(ocr_results) == 1:
                prompts.append("")
                continue

            prompt = self.prompt_template.format(
                language=language,
                context_if_available=contexts[i] or "No additional context available.",
                ocr_result_1=ocr_results[0],
                ocr_result_2=ocr_results[1]
            )
            
            prompts.append(prompt)
            valid_indices.append(i)

        valid_prompts = [p for p in prompts if p]
        
        if not valid_prompts:
            return [results[0] if results else "" for results in ocr_results_list]
        
        try:
            corrected_texts = self.llm_processor.batch_process_with_claude_sync(
                prompts=valid_prompts,
                # system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                concurrency_limit=concurrency_limit
            )

            processed_texts = []
            for response in corrected_texts:
                if "CORRECTED_TEXT:" in response:
                    text = response.split("CORRECTED_TEXT:", 1)[1].strip()
                    processed_texts.append(text)
                else:
                    # If the marker isn't found, add an empty string as a placeholder
                    processed_texts.append("")

            final_results = []
            valid_idx = 0
            
            for i in range(len(ocr_results_list)):
                results = ocr_results_list[i]
                
                if i in valid_indices:
                    text = processed_texts[valid_idx]
                    final_results.append(text if text else (results[0] if results else ""))
                    valid_idx += 1
                else:
                    # Use the first OCR result for skipped items
                    final_results.append(results[0] if results else "")
            
            logger.info(f"Successfully batch-corrected {len(valid_prompts)} OCR texts")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in batch OCR correction: {str(e)}")
            # Fallback to the first OCR result for each input
            return [results[0] if results else "" for results in ocr_results_list]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    test_cases = [
        # Example 1: Two different OCR outputs of the same text with errors
        [
            "〈등장인물〉 이름 : 신유승 전용 특성 : 비스트 테이머 <희귀>, 반성적 살해자 <일반> 전용 스킬 : [길들이기 lv.7]",
            "<등장인물 요약 일람> 이름 : 신유승 전용 특성 :비스트 테이머 (희귀), 반성적 살해자 (일반) 전용 스킬 :[길들이기 Lv.?]"
        ],
        # Example 2: Two variants with punctuation and character differences
        [
            "저는 오늘 아침에 일어났습니다. 해가 밝았고 날씨가 좋았어요.",
            "저는 오늘 아침에 일어났습니다... 해가 밝아았고 날씨가 좋아요!"
        ]
    ]

    corrector = OCRCorrector()
    logger.info("Testing OCR correction with %d samples", len(test_cases))

    for i, ocr_results in enumerate(test_cases):
        logger.info("Test case %d:", i+1)
        for j, text in enumerate(ocr_results):
            logger.info("  OCR %d: %s", j+1, text)

        corrected = corrector.correct_text(ocr_results, language="ko")
        logger.info("  Corrected: %s", corrected)
        
    # Test batch correction
    batch_results = corrector.batch_correct_texts(test_cases, language="ko")
    logger.info("Batch correction results:")
    for i, result in enumerate(batch_results):
        logger.info("  Result %d: %s", i+1, result)