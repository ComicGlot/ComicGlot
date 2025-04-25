"""
LLM Utilities module for ComicGlot.

This module provides utility functions for interacting with Large Language Models,
particularly Claude, for various text processing tasks in the ComicGlot pipeline.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
import asyncio
from dotenv import load_dotenv

# Try to import API clients - install with pip if missing
try:
    from anthropic import Anthropic
except ImportError:
    logging.warning("Anthropic package not installed. Please install with: pip install anthropic")

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Claude model constants
CLAUDE_MODELS = {
    # Newer models
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",  # Most capable model
    "claude-3-5-sonnet-v2": "claude-3-5-sonnet-20241022", 
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",    # Fastest model
    
    # Older models
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-opus": "claude-3-opus-20240229",          # Most capable older model
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",        # Fastest older model
    
    # Latest aliases
    "claude-latest": "claude-3-7-sonnet-20250219",
    "claude-fastest": "claude-3-5-haiku-20241022",
    "claude-balanced": "claude-3-5-sonnet-20241022",
    
    # For specific version control
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229": "claude-3-opus-20240229",
    "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
}

class LLMProcessor:
    """
    Base class for LLM processing operations.
    
    This class provides core functionality for interacting with different LLM APIs
    with a focus on Claude for OCR correction and stylized translation tasks.
    """
    
    def __init__(
        self,
        model_name: str = "claude-balanced",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize the LLM processor.
        
        Args:
            model_name: Name of the model to use (can use friendly names like "claude-latest")
            api_key: API key for the LLM service (if None, read from environment)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation (0.0-1.0)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided for LLM services.")
        
        # Resolve friendly model name to specific model version
        self.model_name = self._resolve_model_name(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize clients if API keys are available
        self._init_clients()
        
    def _resolve_model_name(self, model_name: str) -> str:
        """
        Resolve a friendly model name to a specific model version.
        
        Args:
            model_name: Friendly model name or specific version
            
        Returns:
            Resolved model name with full version
        """
        if model_name.lower() in CLAUDE_MODELS:
            resolved_name = CLAUDE_MODELS[model_name.lower()]
            logger.info(f"Resolved model name '{model_name}' to '{resolved_name}'")
            return resolved_name

        recommended = self.get_recommended_model(model_name.lower())
        resolved_name = CLAUDE_MODELS.get(recommended, recommended)
        logger.info(f"Task-based model '{model_name}' resolved to '{resolved_name}'")
        return resolved_name
      
    
    def _init_clients(self):
        """Initialize API clients based on available API keys."""
        if self.api_key:
            try:
                self.anthropic_client = Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                self.anthropic_client = None
        else:
            self.anthropic_client = None
    
    async def process_with_claude(
        self,
        prompt: str,
        # system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Process text with Claude API.
        
        Args:
            prompt: The prompt to send to Claude
            system_prompt: Optional system prompt to guide Claude's behavior
            model: Model name (overrides instance default if provided)
            max_tokens: Maximum tokens to generate (overrides instance default if provided)
            temperature: Temperature setting (overrides instance default if provided)
            
        Returns:
            Claude's response as text
        
        Raises:
            RuntimeError: If no Anthropic client is available or the request fails
        """
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not available. Check your API key.")

        if model:
            model = self._resolve_model_name(model)
            
        try:
            response = self.anthropic_client.messages.create(
                model=model or self.model_name,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                # system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in Claude API request: {str(e)}")
            raise RuntimeError(f"Failed to get response from Claude: {str(e)}")
    
    def process_with_claude_sync(
        self,
        prompt: str,
        # system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Synchronous version of process_with_claude.
        
        Args:
            prompt: The prompt to send to Claude
            system_prompt: Optional system prompt to guide Claude's behavior
            model: Model name (overrides instance default if provided)
            max_tokens: Maximum tokens to generate (overrides instance default if provided)
            temperature: Temperature setting (overrides instance default if provided)
            
        Returns:
            Claude's response as text
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_with_claude(
                    prompt=prompt,
                    # system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
        finally:
            loop.close()
    
    async def batch_process_with_claude(
        self,
        prompts: List[str],
        # system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        concurrency_limit: int = 5
    ) -> List[str]:
        """
        Process a batch of prompts with Claude API with concurrency control.
        
        Args:
            prompts: List of prompts to send to Claude
            system_prompt: Optional system prompt to guide Claude's behavior
            model: Model name (overrides instance default if provided)
            max_tokens: Maximum tokens to generate (overrides instance default if provided)
            temperature: Temperature setting (overrides instance default if provided)
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            List of Claude's responses as strings
        """
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not available. Check your API key.")

        if model:
            model = self._resolve_model_name(model)
        
        async def process_single(prompt):
            return await self.process_with_claude(
                prompt=prompt,
                # system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def bounded_process(prompt):
            async with semaphore:
                return await process_single(prompt)

        tasks = [bounded_process(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def batch_process_with_claude_sync(
        self,
        prompts: List[str],
        # system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        concurrency_limit: int = 5
    ) -> List[str]:
        """
        Synchronous version of batch_process_with_claude.
        
        Args:
            prompts: List of prompts to send to Claude
            system_prompt: Optional system prompt to guide Claude's behavior
            model: Model name (overrides instance default if provided)
            max_tokens: Maximum tokens to generate (overrides instance default if provided)
            temperature: Temperature setting (overrides instance default if provided)
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            List of Claude's responses as strings
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.batch_process_with_claude(
                    prompts=prompts,
                    # system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    concurrency_limit=concurrency_limit
                )
            )
        finally:
            loop.close()
            
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get a dictionary of available model names and their full version IDs.
        
        Returns:
            Dictionary mapping friendly model names to full version IDs
        """
        return CLAUDE_MODELS.copy()
    
    @staticmethod
    def get_recommended_model(task: str = 'general') -> str:
        """
        Get the recommended model for a specific task.
        
        Args:
            task: Type of task ('general', 'ocr_correction', 'translation', 'fast')
            
        Returns:
            Recommended model name
        """
        recommendations = {
            'general': 'claude-balanced',
            'ocr_correction': 'claude-3-7-sonnet',  # OCR needs high accuracy
            'translation': 'claude-3-7-sonnet',       # Translation needs balance of speed/capability
            'fast': 'claude-fastest',               # When speed is most important
            'quality': 'claude-3-7-sonnet',         # When quality is most important
            'cheap': 'claude-3-haiku',              # When cost is most important
        }
        return recommendations.get(task.lower(), 'claude-balanced')