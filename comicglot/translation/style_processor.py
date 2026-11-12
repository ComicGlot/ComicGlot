"""
Style Processor module for ComicGlot.

This module provides functionality to perform stylized translation of text
in comics based on the bubble type, maintaining the emotional and stylistic 
elements of the original text.
"""

import os
import json
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

class StyleTranslator:
    """
    Style Translator class that performs context-aware, stylized translation
    of comic text based on bubble types and surrounding context.
    """
    
    # Mismatched guide per class for the wrong-class ablation; mirrors
    # eval/translation_study/run_conditions.py so pipeline runs and study runs agree.
    WRONG_CLASS = {
        "speech_bubble": "scream_bubble", "scream_bubble": "think_bubble", "think_bubble": "scream_bubble",
        "cloud_bubble": "spiked_bubble", "spiked_bubble": "cloud_bubble", "wiggle_bubble": "scream_bubble",
        "hexagon_bubble": "wiggle_bubble", "sqare_bubble": "scream_bubble", "system_bubble": "wiggle_bubble",
        "standalone_text": "system_bubble", "background_text": "scream_bubble", "title_panel": "think_bubble",
        "sfx_sound": "speech_bubble",
    }

    @classmethod
    def guide_class(cls, bubble_type: str, condition: str = "true") -> str:
        """Class whose style guide is used: the detected one, none, or a mismatched one."""
        if condition == "true":
            return bubble_type
        if condition == "none":
            return "default"
        if condition == "shuffled":
            return cls.WRONG_CLASS.get(bubble_type, bubble_type)
        raise ValueError(condition)

    # The detector's label set spells the square class "sqare_bubble" while the
    # style guides use "square_bubble". Before this alias, square blocks silently
    # received the default guide (21 of 516 blocks in the chapter 3 and 4 runs).
    CLASS_ALIASES = {"sqare_bubble": "square_bubble"}

    def _guide_for(self, bubble_type: str) -> str:
        key = self.CLASS_ALIASES.get(bubble_type, bubble_type)
        return self.style_guides.get(key, self.style_guides.get("default", self.DEFAULT_FALLBACK))

    DEFAULT_FALLBACK = (
        "Translate this text naturally while preserving the original tone and meaning. "
        "Consider context and maintain any emphasis or special formatting from the original."
    )
    
    def __init__(
        self,
        llm_processor: Optional[LLMProcessor] = None,
        model_name: str = "translation",
        # system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        style_guides_path: str = "style_guides.json",
        prompt_variant: Optional[str] = None,
    ):
        """
        Initialize the style translator.
        
        Args:
            llm_processor: LLMProcessor instance (if None, creates a new one)
            model_name: Name of the LLM model to use
            system_prompt: System prompt to guide the LLM's behavior
            temperature: Temperature parameter for LLM generation
            style_guides_path: Path to JSON file containing style guides (defaults to style_guides.json)
        """
        self.llm_processor = llm_processor or LLMProcessor(
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )
        
        self.system_prompt = None
        # self.system_prompt = system_prompt or (
        #     "You are an expert comics translator specializing in adaptive, context-aware translation "
        #     "that preserves the style, emotion, and narrative impact of the original text. "
        #     "Your translations should feel natural in the target language while maintaining "
        #     "the distinctive style indicated by different types of speech bubbles and text formats. "
        #     "Pay special attention to idioms, cultural references, and sound effects, adapting them "
        #     "appropriately for the target audience."
        # )

        self.style_guides = self._load_style_guides(style_guides_path)
        # A prompt variant of the prompt lab (data/prompts/lab/<name>.json): its own template and guides. None keeps
        # the published prompt and style_guides.json.
        self.prompt_variant = prompt_variant
        self.variant = self._load_variant(prompt_variant) if prompt_variant else None
        self.temperature = temperature
        if self.variant and "temperature" in self.variant:  # a variant's temperature is part of its design
            self.temperature = float(self.variant["temperature"])
        logger.info(f"Style Translator initialized with {len(self.style_guides)} style guides")

        self.prompt_template = (
            "[TASK]: Translate text from {source_lang} to {target_lang} with appropriate style.\n\n"
            "{context_section}"
            "[ORIGINAL TEXT]: {text}\n\n"
            "[STYLE GUIDE]: {style_guide}\n\n"
            "[INSTRUCTIONS]: Translate the original text from {source_lang} to {target_lang} "
            "following the style guide appropriate for a {bubble_type}.\n\n"
            "Analyze the original text, taking into account any context provided and the style guide. "
            "Think carefully about cultural nuances, idioms, and emotional tone.\n\n"
            "After your analysis, provide ONLY the translated text with no additional explanation, "
            "starting with 'TRANSLATED_TEXT:' followed by the translation."
        )
    
    @staticmethod
    def variant_path(name: str) -> str:
        if os.path.exists(name):
            return name
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join("data", "prompts", "lab", f"{name}.json"),
                  os.path.join(here, "..", "..", "data", "prompts", "lab", f"{name}.json")):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"prompt variant {name!r} not found in data/prompts/lab")

    def _load_variant(self, name: str) -> Dict:
        with open(self.variant_path(name), encoding="utf-8") as f:
            v = json.load(f)
        logger.info(f"Prompt variant {name}: {len(v.get('guides', {}))} guides")
        return v

    def _build_prompt(self, text: str, source_lang: str, target_lang: str, bubble_type: str, context) -> str:
        """The published prompt, or the prompt variant's. With a variant, bubble_type "default" (the class withheld)
        gives the variant's plain prompt: the same template without the type and its guide."""
        if not self.variant:
            return self.prompt_template.format(
                source_lang=source_lang, target_lang=target_lang, text=text,
                context_section=self._format_context_section(context),
                style_guide=self._guide_for(bubble_type), bubble_type=bubble_type)
        v = self.variant
        key = self.CLASS_ALIASES.get(bubble_type, bubble_type)
        guided = key in v["types"]
        if v.get("guided_types") and key not in v["guided_types"]:
            guided = False  # selective guides: this type is translated with the plain prompt
        lines = []
        for group in context or []:
            for b in group.get("bubbles", []):
                if b.get("text"):
                    tag = f" ({b.get('bubble_type')})" if guided and v.get("context_types", True) else ""
                    lines.append(f"- Prev text {len(lines) + 1}{tag}: {b['text']}")
        ctx = "[CONTEXT]: Surrounding texts:\n" + "\n".join(lines) + "\n\n" if lines else ""
        typ = v["type_line"].format(type=v["types"][key]) if guided and v.get("type_line") else ""
        guide = v["guide_line"].format(guide=v["guides"][key]) if guided and v["guides"].get(key) else v.get("plain_guide_line", "") if not guided else ""
        return v["template"].format(text=text, context=ctx, type=typ, guide=guide).replace("\n\n\n", "\n\n")

    def _load_style_guides(self, path: str) -> Dict[str, str]:
        """
        Load style guides from the style_guides.json file.
        
        Args:
            path: Path to style_guides.json file
            
        Returns:
            Dictionary of style guides for different bubble types
        """
        guides = {"default": self.DEFAULT_FALLBACK}
        
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    loaded_guides = json.load(f)
                logger.info(f"Loaded {len(loaded_guides)} style guides from {path}")
                return loaded_guides

            default_locations = [
                "style_guides.json",
                os.path.join("data", "prompts", "style_guides.json"),
                os.path.join("config", "style_guides.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "style_guides.json")
            ]
            
            for loc in default_locations:
                if os.path.exists(loc):
                    with open(loc, 'r', encoding='utf-8') as f:
                        loaded_guides = json.load(f)
                    logger.info(f"Loaded {len(loaded_guides)} style guides from {loc}")
                    return loaded_guides

            logger.warning(f"Style guides file not found at {path} or any default locations")
            logger.warning("Creating a new style_guides.json file with default styles")

            default_guides = {
                "speech_bubble": "This text is from a standard speech bubble. Use natural, conversational language appropriate for dialogue. Maintain the original tone and any slang or informal expressions in an equivalent way in the target language.",
                "cloud_bubble": "This text is from a cloud-shaped bubble, indicating upbeat or happy speech. Use a light, cheerful tone. Include appropriate elongated words or sounds to convey happiness or excitement. Use tildes (~) where appropriate for emphasis.",
                "scream_bubble": "This text is from a scream bubble with jagged edges, indicating shouting or intense emotion. Use multiple exclamation marks and strong emotional language. Emphasize the intensity of the feeling being expressed.",
                "spiked_bubble": "This text is from a spiked bubble indicating intense or aggressive thoughts. Use sharp, direct language with strong emphasis. May include internal conflict or intense emotions.",
                "system_bubble": "This text is system information or UI text. Use a formal, precise style. Maintain any technical terms, status indicators, or formatting. Keep it clean and structured, similar to video game interface text.",
                "think_bubble": "This text represents the character's thoughts. Use introspective, reflective language. Include thought fragments, ellipses (...) where appropriate. The tone should feel internal and contemplative.",
                "standalone_text": "This is standalone text without a bubble. Use a neutral narrative voice. It may be background information or scene-setting text. Keep the translation concise.",
                "sfx_sound": "This text represents a sound effect. Instead of literal translation, use the equivalent sound effect in the target language that would convey the same sound or impact. If the target language doesn't have an equivalent, adapt the sound effect to fit the phonetic patterns of the target language. Use ALL CAPS for emphasis where appropriate.",
                "hexagon_bubble": "This text is from a hexagonal bubble, often indicating mechanical or magical speech. Use a more formal or structured tone, possibly with specialized vocabulary depending on the context.",
                "square_bubble": "This text is from a square bubble, often used for narration or voice-overs. Use a clear, narrative style. This may be more formal than regular dialogue.",
                "wiggle_bubble": "This text is from a wavy/wiggling bubble, indicating weakness, fear, or uncertainty. Use stuttering, trembling speech patterns, or hesitant language. May include ellipses and broken sentence structures to convey the emotional state.",
                "title_panel": "This is title or chapter text. Translate with emphasis on clarity and impact. Consider cultural context for chapter titles, maintaining any wordplay or special meaning if possible. Format should match the original's emphasis.",
                "background_text": "This is background text integrated into the scene, like signs, books, or environmental text. Translate with attention to physical context. Keep it concise and appropriate to its location in the image.",
                "default": self.DEFAULT_FALLBACK
            }

            try:
                directory = os.path.dirname(path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(default_guides, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Created new style guides file at {path}")
                return default_guides
            except Exception as e:
                logger.error(f"Failed to create style guides file: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error loading style guides: {str(e)}")
            logger.warning("Using only the default fallback style guide")
        
        return guides

    def _format_context_section(self, context=None):
        """
        Format context section for prompt template based on structured JSON context.
        
        Args:
            context: JSON-formatted context containing strategy and bubble information
            
        Returns:
            Formatted context section string
        """
        if not context:
            return ""
        
        context_parts = []
        
        # Process each context group (could be prev/next/mid)
        for context_group in context:
            strategy = context_group.get("context_strategy", "unknown")
            bubbles = context_group.get("bubbles", [])
            
            if not bubbles:
                continue

            group_lines = []
            for idx, bubble in enumerate(bubbles):
                text = bubble.get("text", "")
                bubble_type = bubble.get("bubble_type", "unknown")

                if not text:
                    continue
                    
                group_lines.append(f"- {strategy.capitalize()} text {idx+1} ({bubble_type}): {text}")
            
            if group_lines:
                context_parts.append("\n".join(group_lines))
        
        if not context_parts:
            return ""

        all_context = "\n\n".join(context_parts)
        return f"[CONTEXT]: Surrounding texts:\n{all_context}\n\n"

    def _generate_batch_context(self, texts: List[str], bubble_types: List[str], 
                                current_index: int, context_strategy: str, window_size: int = 2):
        """
        Generate context for a specific bubble in a batch based on strategy.
        
        Args:
            texts: List of all texts in the batch
            bubble_types: List of all bubble types in the batch
            current_index: Index of the current bubble being processed
            context_strategy: Strategy for context selection ("prev", "next", "mid")
            window_size: Number of context bubbles to include (default: 2)
            
        Returns:
            List of context group objects
        """
        if not texts or not bubble_types or current_index < 0 or current_index >= len(texts):
            return []
            
        context = []
        
        if context_strategy == "prev":
            prev_indices = range(max(0, current_index - window_size), current_index)
            if prev_indices:
                prev_bubbles = [
                    {"text": texts[i], "bubble_type": bubble_types[i]} 
                    for i in prev_indices
                ]
                if prev_bubbles:
                    context.append({
                        "context_strategy": "prev",
                        "bubbles": prev_bubbles
                    })
                    
        elif context_strategy == "next":
            next_indices = range(current_index + 1, min(len(texts), current_index + window_size + 1))
            if next_indices:
                next_bubbles = [
                    {"text": texts[i], "bubble_type": bubble_types[i]} 
                    for i in next_indices
                ]
                if next_bubbles:
                    context.append({
                        "context_strategy": "next",
                        "bubbles": next_bubbles
                    })
                
        elif context_strategy == "mid":
            # Try to position the current bubble in the middle of the context window
            # For first bubbles, use more next bubbles; for last bubbles, use more previous ones
            
            total_bubbles = len(texts)
            if current_index < window_size / 2:
                prev_count = current_index
                next_count = min(window_size - prev_count, total_bubbles - current_index - 1)
            elif current_index >= total_bubbles - window_size / 2:
                next_count = total_bubbles - current_index - 1
                prev_count = min(window_size - next_count, current_index)
            else:
                prev_count = window_size // 2
                next_count = window_size - prev_count
            
            if prev_count > 0:
                prev_indices = range(current_index - prev_count, current_index)
                prev_bubbles = [
                    {"text": texts[i], "bubble_type": bubble_types[i]} 
                    for i in prev_indices
                ]
                if prev_bubbles:
                    context.append({
                        "context_strategy": "prev",
                        "bubbles": prev_bubbles
                    })
            
            if next_count > 0:
                next_indices = range(current_index + 1, current_index + next_count + 1)
                next_bubbles = [
                    {"text": texts[i], "bubble_type": bubble_types[i]} 
                    for i in next_indices if i < total_bubbles
                ]
                if next_bubbles:
                    context.append({
                        "context_strategy": "next",
                        "bubbles": next_bubbles
                    })
                    
        return context
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        bubble_type: str = "speech_bubble",
        context: Optional[List[Dict]] = None,
        max_tokens: int = 1024
    ) -> str:
        """
        Translate text with stylistic adaptation based on bubble type.
        
        Args:
            text: Text to translate
            source_lang: Source language ISO code
            target_lang: Target language ISO code
            bubble_type: Type of bubble containing the text
            context: Optional context (list of context group objects)
            max_tokens: Maximum tokens for LLM response
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""

        prompt = self._build_prompt(text, source_lang, target_lang, bubble_type, context)
        
        try:
            logger.debug(f"Sending translation prompt to LLM for bubble type {bubble_type}")
            response = self.llm_processor.process_with_claude_sync(
                prompt=prompt,
                # system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=self.temperature
            )

            if "TRANSLATED_TEXT:" in response:
                translated_text = response.split("TRANSLATED_TEXT:", 1)[1].strip()
                logger.info(f"Successfully translated text ({bubble_type})")
                return translated_text
            else:
                logger.warning("LLM response did not contain 'TRANSLATED_TEXT:' marker")
                logger.debug(f"Raw LLM response: {response}")
                return response.strip()
            
        except Exception as e:
            logger.error(f"Error in translation: {str(e)}")
            return text
    
    async def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        bubble_types: List[str],
        contexts: Optional[List[List[Dict]]] = None,
        context_strategy: Optional[str] = None,
        context_window_size: int = 2,
        max_tokens: int = 1024,
        concurrency_limit: int = 5
    ) -> List[str]:
        """
        Translate multiple texts in batch with flexible context handling.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language ISO code
            target_lang: Target language ISO code
            bubble_types: List of bubble types for each text
            contexts: Optional list of context objects for each text
            context_strategy: Strategy for auto-generating context ("prev", "next", "mid")
            context_window_size: Size of the context window when using auto-generation
            max_tokens: Maximum tokens for LLM response
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            List of translated texts
        """
        if not texts:
            logger.warning("Empty list of texts provided for translation")
            return []
        
        if len(texts) != len(bubble_types):
            logger.error("Number of texts must match number of bubble types")
            raise ValueError("Number of texts must match number of bubble types")

        # Handle context generation strategies
        if contexts is not None and context_strategy is not None:
            logger.error("Cannot provide both 'contexts' and 'context_strategy' parameters")
            raise ValueError("Cannot provide both 'contexts' and 'context_strategy' parameters")

        if context_strategy is not None:
            if context_strategy not in ["prev", "next", "mid"]:
                logger.error(f"Invalid context_strategy: {context_strategy}")
                raise ValueError("context_strategy must be one of: 'prev', 'next', 'mid'")

            logger.info(f"Generating context using '{context_strategy}' strategy")
            contexts = []
            for i in range(len(texts)):
                ctx = self._generate_batch_context(
                    texts=texts,
                    bubble_types=bubble_types,
                    current_index=i,
                    context_strategy=context_strategy,
                    window_size=context_window_size
                )
                contexts.append(ctx)

        if contexts is None:
            contexts = [None] * len(texts)
        elif len(contexts) != len(texts):
            logger.error(f"Number of contexts ({len(contexts)}) does not match number of texts ({len(texts)})")
            raise ValueError("Number of contexts must match number of texts")

        logger.info(f"Preparing translation batch with {len(texts)} texts")

        prompts = []
        valid_indices = []
        
        for i, (text, bubble_type) in enumerate(zip(texts, bubble_types)):
            if not text.strip():
                prompts.append("")  # Empty prompt for empty text
                logger.debug(f"Skipping empty text at index {i}")
                continue

            prompt = self._build_prompt(text, source_lang, target_lang, bubble_type, contexts[i])
            
            prompts.append(prompt)
            valid_indices.append(i)
            logger.debug(f"Added text {i} ({bubble_type}) to translation batch: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        valid_prompts = [p for p in prompts if p]
        filtered_indices = [valid_indices[i] for i, p in enumerate(prompts) if p]
        
        if not valid_prompts:
            logger.warning("No valid translation prompts after filtering")
            return texts  # Return original texts if no valid prompts
        
        try:
            all_responses = await self.llm_processor.batch_process_with_claude( # batch_process_with_claude_sync
                prompts=valid_prompts,
                max_tokens=max_tokens,
                temperature=self.temperature,
                #concurrency_limit=concurrency_limit
            )

            translated_texts = []
            for i, response in enumerate(all_responses):
                if "TRANSLATED_TEXT:" in response:
                    translated_text = response.split("TRANSLATED_TEXT:", 1)[1].strip()
                    translated_texts.append(translated_text)
                    logger.debug(f"Translation {i} successful")
                else:
                    logger.warning(f"Translation {i} response missing 'TRANSLATED_TEXT:' marker: {response[:100]}...")
                    translated_texts.append(response.strip())

            final_results = texts.copy()
            for i, idx in enumerate(filtered_indices):
                if i < len(translated_texts):
                    final_results[idx] = translated_texts[i]
                else:
                    logger.warning(f"Missing translation for text at index {idx}")
            
            logger.info(f"Successfully batch-translated {len(valid_prompts)} texts")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return texts
    
    @staticmethod
    def save_style_guides(style_guides: Dict[str, str], filepath: str) -> bool:
        """
        Save style guides to a JSON configuration file.
        
        Args:
            style_guides: Dictionary of style guides
            filepath: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(style_guides, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Saved {len(style_guides)} style guides to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving style guides to {filepath}: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    test_texts = [
        "안녕하세요! 오늘 어떻게 지내세요?",  # Hello! How are you doing today?
        "이건 정말 큰 문제가 될 수 있어...",  # This could become a really big problem...
        "으아아아! 조심해!!!",               # Aaaaah! Be careful!!!
        "그날, 세상은 변했다.",              # That day, the world changed.
        "쾅! 쿵! 찌지직!",                  # Boom! Thud! Crackle!
        "...네가 여기 있을 줄은 몰랐어..."    # ...I didn't know you'd be here...
    ]

    test_bubble_types = [
        "speech_bubble",
        "think_bubble",
        "scream_bubble",
        "square_bubble",
        "sfx_sound",
        "wiggle_bubble"
    ]

    translator = StyleTranslator()

    logger.info("Testing individual translation:")
    for i, (text, bubble_type) in enumerate(zip(test_texts, test_bubble_types)):
        logger.info(f"Test {i+1} - {bubble_type}:")
        logger.info(f"  Original: {text}")
        translated = translator.translate(
            text=text,
            source_lang="ko",
            target_lang="en",
            bubble_type=bubble_type
        )
        logger.info(f"  Translated: {translated}")
    

    # Test with "mid" strategy
    logger.info("\nTesting batch translation with automatic context (mid strategy):")
    batch_results_mid = translator.batch_translate(
        texts=test_texts,
        source_lang="ko",
        target_lang="en",
        bubble_types=test_bubble_types,
        context_strategy="mid",
        context_window_size=2
    )
    
    for i, (original, translated, bubble_type) in enumerate(zip(test_texts, batch_results_mid, test_bubble_types)):
        logger.info(f"Batch result with auto context (mid) {i+1} - {bubble_type}:")
        logger.info(f"  Original: {original}")
        logger.info(f"  Translated: {translated}")