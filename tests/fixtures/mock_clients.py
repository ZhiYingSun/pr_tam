"""
Mock client implementations for testing purposes.
Implements client protocols without making actual API calls.
"""
import asyncio
import base64
import json
import logging
from typing import Dict, Any, Optional, List

from src.models.api_models import ZyteHttpResponse
from src.clients.client_protocols import ZyteClientProtocol, OpenAIClientProtocol

logger = logging.getLogger(__name__)


class MockZyteClient(ZyteClientProtocol):
    """
    Mock implementation of ZyteClientProtocol for testing.
    
    Returns predefined responses without making actual HTTP requests.
    """
    
    def __init__(self, api_key: str = "mock_key"):
        """
        Initialize mock Zyte client.
        
        Args:
            api_key: Mock API key (not used, but maintained for interface compatibility)
        """
        self.api_key = api_key
        self.post_call_count = 0
        self.get_call_count = 0
        self.post_requests: List[Dict[str, Any]] = []
        self.get_requests: List[str] = []
        
        # Predefined mock responses
        self.mock_search_response = {
            "response": {
                "records": [
                    {
                        "businessEntityId": 12345,
                        "registrationNumber": 12345,
                        "registrationIndex": "12345-111",
                        "corpName": "Test Restaurant Corp",
                        "statusEn": "ACTIVE",
                        "statusEs": "ACTIVA"
                    },
                    {
                        "businessEntityId": 12346,
                        "registrationNumber": 12346,
                        "registrationIndex": "12346-222",
                        "corpName": "Test Restaurant LLC",
                        "statusEn": "ACTIVE",
                        "statusEs": "ACTIVA"
                    }
                ]
            },
            "code": 1,
            "info": None,
            "success": True
        }
        
        self.mock_detail_response = {
            "response": {
                "corporation": {
                    "businessEntityId": 12345,
                    "registrationNumber": 12345,
                    "registrationIndex": "12345-111",
                    "corpName": "Test Restaurant Corp",
                    "statusEn": "ACTIVE",
                    "statusEs": "ACTIVA"
                },
                "mainLocation": {
                    "streetAddress": {
                        "address1": "123 Main St",
                        "address2": None,
                        "city": "San Juan",
                        "zip": "00901"
                    }
                },
                "residentAgent": {
                    "isIndividual": True,
                    "individualName": {
                        "firstName": "John",
                        "middleName": "A",
                        "lastName": "Doe",
                        "surName": None
                    },
                    "organizationName": None,
                    "streetAddress": {
                        "address1": "456 Agent St",
                        "address2": None
                    }
                }
            },
            "code": 1,
            "success": True
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        logger.debug("MockZyteClient: entering context")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.debug("MockZyteClient: exiting context")
        pass
    
    async def close(self):
        """Close the client session (no-op for mock)."""
        logger.debug("MockZyteClient: close called")
        pass
    
    async def post_request(
        self,
        url: str,
        request_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        """
        Mock POST request - returns predefined search response.
        """
        # Simulate async delay
        await asyncio.sleep(0.001)
        
        # Track the request
        self.post_call_count += 1
        self.post_requests.append({
            "url": url,
            "body": request_body,
            "headers": headers
        })
        
        logger.debug(f"MockZyteClient: POST request #{self.post_call_count} to {url}")
        
        # Encode mock response as base64 (mimicking Zyte's response format)
        response_json = json.dumps(self.mock_search_response)
        response_base64 = base64.b64encode(response_json.encode('utf-8')).decode('utf-8')
        
        return ZyteHttpResponse(httpResponseBody=response_base64)
    
    async def get_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> ZyteHttpResponse:
        """
        Mock GET request - returns predefined detail response.
        """
        # Simulate async delay
        await asyncio.sleep(0.001)
        
        # Track the request
        self.get_call_count += 1
        self.get_requests.append(url)
        
        logger.debug(f"MockZyteClient: GET request #{self.get_call_count} to {url}")
        
        # Encode mock response as base64
        response_json = json.dumps(self.mock_detail_response)
        response_base64 = base64.b64encode(response_json.encode('utf-8')).decode('utf-8')
        
        return ZyteHttpResponse(httpResponseBody=response_base64)
    
    def reset_tracking(self):
        """Reset call tracking counters and logs."""
        self.post_call_count = 0
        self.get_call_count = 0
        self.post_requests = []
        self.get_requests = []
    
    def set_search_response(self, response: Dict[str, Any]):
        """Override the default search response with custom data."""
        self.mock_search_response = response
    
    def set_detail_response(self, response: Dict[str, Any]):
        """Override the default detail response with custom data."""
        self.mock_detail_response = response


class MockOpenAIClient(OpenAIClientProtocol):
    """
    Mock implementation of OpenAIClientProtocol for testing.
    
    Returns predefined responses without making actual API calls.
    """
    
    def __init__(self, api_key: str = "mock_key"):
        """
        Initialize mock OpenAI client.
        
        Args:
            api_key: Mock API key (not used, but maintained for interface compatibility)
        """
        self.api_key = api_key
        self.call_count = 0
        self.requests: List[Dict[str, Any]] = []
        
        # Default mock response for validation
        self.mock_response = {
            "is_match": True,
            "confidence": "high",
            "reasoning": "Mock validation - exact name and address match"
        }
    
    async def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Mock chat completion - returns predefined response.
        """
        # Simulate async delay
        await asyncio.sleep(0.001)
        
        # Track the request
        self.call_count += 1
        self.requests.append({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format
        })
        
        logger.debug(f"MockOpenAIClient: chat_completion call #{self.call_count} with model {model}")
        
        # Return the mock response
        return self.mock_response
    
    def reset_tracking(self):
        """Reset call tracking counters and logs."""
        self.call_count = 0
        self.requests = []
    
    def set_response(self, response: Dict[str, Any]):
        """Override the default response with custom data."""
        self.mock_response = response

