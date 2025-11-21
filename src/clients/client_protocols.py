"""
Protocols (interfaces) for client implementations.
Allows for dependency injection and easier testing.
"""
from typing import Protocol, Dict, Any, Optional
from src.models.api_models import ZyteHttpResponse


class ZyteClientProtocol(Protocol):
    """
    Protocol defining the interface for Zyte API clients.
    
    This allows for dependency injection where we can swap
    real implementations with mock implementations for testing.
    """
    
    async def __aenter__(self):
        """Async context manager entry."""
        ...
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        ...
    
    async def close(self):
        """Close the client session."""
        ...
    
    async def post_request(
        self,
        url: str,
        request_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        """
        Make a POST request.
        
        Args:
            url: Target URL
            request_body: Request payload
            headers: Optional HTTP headers
            
        Returns:
            ZyteHttpResponse object
        """
        ...
    
    async def get_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        """
        Make a GET request.
        
        Args:
            url: Target URL
            headers: Optional HTTP headers
            
        Returns:
            ZyteHttpResponse object
        """
        ...


class OpenAIClientProtocol(Protocol):
    """
    Protocol defining the interface for OpenAI API clients.
    
    This allows for dependency injection where we can swap
    real implementations with mock implementations for testing.
    """
    
    async def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make a chat completion request to OpenAI API.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Parsed JSON response as dict, or None on error
        """
        ...

