"""
Zyte API client - Singleton async client for making requests through Zyte
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import ClientSession, ClientTimeout, BasicAuth

from src.data.models import ZyteHttpResponse

logger = logging.getLogger(__name__)


class ZyteClient:
    """
    Singleton async Zyte API client with connection pooling.
    Ensures all instances share the same connection pool for efficiency.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, api_key: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, api_key: str = None):
        if not self._initialized:
            self.api_key = api_key or os.getenv('ZYTE_API_KEY')
            if not self.api_key:
                raise ValueError("Zyte API key must be provided or set in ZYTE_API_KEY environment variable")
            self.session = None
            self._initialized = True
    
    async def __aenter__(self):
        """Async context manager entry - create session with connection pooling."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=20,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            timeout = ClientTimeout(total=60, connect=10)
            self.session = ClientSession(
                connector=connector,
                timeout=timeout,
                auth=BasicAuth(self.api_key, "")
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session if needed."""
        # Don't close session here - keep it alive for reuse
        # Session will be closed when explicitly needed or on shutdown
        pass
    
    async def close(self):
        """Explicitly close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def post_request(
        self,
        url: str,
        request_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        if not self.session or self.session.closed:
            raise RuntimeError("ZyteClient session not initialized. Use 'async with ZyteClient(...)'")
        
        payload = {
            "url": url,
            "httpResponseBody": True,
            "httpRequestMethod": "POST",
            "httpRequestText": json.dumps(request_body),
        }
        
        if headers:
            payload["customHttpRequestHeaders"] = [
                {"name": k, "value": v} for k, v in headers.items()
            ]
        
        try:
            async with self.session.post(
                "https://api.zyte.com/v1/extract",
                json=payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Zyte API returned status {resp.status}")
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status
                    )
                
                zyte_data = await resp.json()
                
                # Parse with Pydantic and decode
                zyte_response = ZyteHttpResponse(**zyte_data)
                return zyte_response
                
        except (ValueError, aiohttp.ClientError) as e:
            logger.error(f"Zyte POST request failed: {e}")
            raise
    
    async def get_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        """
        Make a GET request through Zyte API.
        
        Args:
            url: Target URL to request
            headers: Optional custom headers
            
        Returns:
            Decoded response body as dict
            
        Raises:
            ValueError: If response decoding fails
            aiohttp.ClientError: If HTTP request fails
        """
        if not self.session or self.session.closed:
            raise RuntimeError("ZyteClient session not initialized. Use 'async with ZyteClient(...)'")
        
        payload = {
            "url": url,
            "httpResponseBody": True,
            "httpRequestMethod": "GET",
        }
        
        if headers:
            payload["customHttpRequestHeaders"] = [
                {"name": k, "value": v} for k, v in headers.items()
            ]
        
        try:
            async with self.session.post(
                "https://api.zyte.com/v1/extract",
                json=payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Zyte API returned status {resp.status}")
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status
                    )
                
                zyte_data = await resp.json()
                
                # Parse with Pydantic and decode
                zyte_response = ZyteHttpResponse(**zyte_data)
                return zyte_response
                
        except (ValueError, aiohttp.ClientError) as e:
            logger.error(f"Zyte GET request failed: {e}")
            raise
