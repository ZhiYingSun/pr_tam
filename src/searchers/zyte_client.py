"""
Zyte API client for making requests through Zyte
"""
import json
import asyncio
from base64 import b64decode
from typing import Dict, Any, Optional
import logging
from aiohttp import ClientSession, ClientTimeout, BasicAuth

logger = logging.getLogger(__name__)


async def post_request(
    api_key: str,
    url: str,
    request_body: dict[str, any],
    headers: dict[str, any] | None = None,
) -> dict:
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

    async with ClientSession(timeout=ClientTimeout(total=60)) as session:
        async with session.post(
            "https://api.zyte.com/v1/extract",
            auth=BasicAuth(api_key, ""),
            json=payload,
        ) as resp:
            data = await resp.json()
            body = json.loads(b64decode(data["httpResponseBody"]))
            return body


async def get_request(
    api_key: str,
    url: str,
    headers: dict[str, any] | None = None,
) -> str:
    payload = {
        "url": url,
        "httpResponseBody": True,
        "httpRequestMethod": "GET",
    }

    if headers:
        payload["customHttpRequestHeaders"] = [
            {"name": k, "value": v} for k, v in headers.items()
        ]

    async with ClientSession(timeout=ClientTimeout(total=60)) as session:
        async with session.post(
            "https://api.zyte.com/v1/extract",
            auth=BasicAuth(api_key, ""),
            json=payload,
        ) as resp:
            data = await resp.json()
            decoded = b64decode(data["httpResponseBody"]).decode("utf-8", errors="ignore")
            return decoded


class ZyteClient:
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def post_request_sync(
        self,
        url: str,
        request_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        try:
            return asyncio.run(post_request(self.api_key, url, request_body, headers))
        except Exception as e:
            logger.error(f"Zyte POST request failed: {e}")
            return {}
    
    def get_request_sync(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        try:
            return asyncio.run(get_request(self.api_key, url, headers))
        except Exception as e:
            logger.error(f"Zyte GET request failed: {e}")
            return ""
