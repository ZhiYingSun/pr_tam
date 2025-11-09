"""
OpenAI API client
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from openai import APIStatusError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Singleton async OpenAI API client with rate limiting and no business logic.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, api_key: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, api_key: str = None):
        if not self._initialized:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
            # Rate limiter: 500 requests per minute
            self.rate_limiter = AsyncLimiter(max_rate=500, time_period=60)
            
            # Initialize OpenAI client
            self.client = AsyncOpenAI(api_key=self.api_key)
            
            self._initialized = True
            logger.info("Initialized OpenAI client with rate limiter (500/minute)")
    
    async def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make a chat completion request to OpenAI API with rate limiting.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Parsed JSON response as dict, or None on error
        """
        async with self.rate_limiter:
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content
                if response_format and response_format.get("type") == "json_object":
                    return json.loads(content)
                return {"content": content}
                
            except APIStatusError as e:
                logger.error(f"OpenAI API error (status {e.status_code}): {e.response}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from OpenAI response: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI API call: {e}")
                return None

