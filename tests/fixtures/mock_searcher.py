"""
Mock searcher for testing purposes.
Returns predefined data without making actual API calls.
"""
import asyncio
import re
import logging
from typing import List

from src.models.models import BusinessRecord

logger = logging.getLogger(__name__)


class AsyncMockIncorporationSearcher:
    """
    Async mock implementation of IncorporationSearcher for testing purposes.
    Returns predefined data without making actual API calls.
    """
    
    def __init__(self, zyte_api_key: str = "mock_key"):
        self.api_key = zyte_api_key
        self.mock_search_results = {
            "Test Restaurant": [
                {"registrationNumber": 123451, "corpName": "Test Restaurant Corp 1", "statusEn": "ACTIVE"},
                {"registrationNumber": 123452, "corpName": "Test Restaurant Corp 2", "statusEn": "ACTIVE"},
                {"registrationNumber": 123453, "corpName": "Test Restaurant Corp 3", "statusEn": "INACTIVE"},
            ],
            "Condal Tapas Restaurant & Rooftop Lounge": [
                {"registrationNumber": 123451, "corpName": "Condal Tapas Restaurant & Rooftop Lounge Corp 1", "statusEn": "ACTIVE"},
                {"registrationNumber": 123452, "corpName": "Condal Tapas Restaurant & Rooftop Lounge Corp 2", "statusEn": "ACTIVE"},
                {"registrationNumber": 123453, "corpName": "Condal Tapas Restaurant & Rooftop Lounge Corp 3", "statusEn": "ACTIVE"},
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def search_business_async(self, business_name: str, limit: int = 5) -> List[BusinessRecord]:
        """
        Mocks searching for businesses asynchronously.
        Returns a list of BusinessRecord objects based on predefined mock data.
        """
        # Simulate async delay
        await asyncio.sleep(0.001)  # 1ms delay to simulate real async behavior
        
        # Try to find matches for normalized names
        mock_records = []
        
        # Check original name first
        if business_name in self.mock_search_results:
            mock_records = self.mock_search_results[business_name]
        else:
            # Check normalized versions of the keys
            for key, records in self.mock_search_results.items():
                normalized_key = self._normalize_name(key)
                if normalized_key == business_name:
                    mock_records = records
                    break
        
        business_records = []
        for record in mock_records[:limit]:
            try:
                business_record = BusinessRecord(
                    legal_name=record.get('corpName', ''),
                    registration_number=str(record.get('registrationNumber', '')),
                    registration_index='',  # Mock data
                    status=record.get('statusEn', ''),
                    # Optional fields left as None for mock data
                )
                business_records.append(business_record)
            except Exception as e:
                logger.warning(f"Failed to create mock business record: {e}")
                continue
        
        return business_records
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching (same logic as RestaurantMatcher)"""
        if not name or not isinstance(name, str):
            return ""
            
        name = name.lower()
        
        # Remove common suffixes
        common_suffixes = [
            "llc", "inc", "corp", "ltd", "co", "restaurant", "bar", "cafe", "grill",
            "eats", "kitchen", "pub", "diner", "bistro", "pizzeria", "cantina",
            "taqueria", "bakery", "store", "market", "shop", "supercenter", "supermarket"
        ]
        
        for suffix in common_suffixes:
            name = re.sub(r'\b' + re.escape(suffix) + r'\b', '', name)
        
        # Remove punctuation
        name = re.sub(r'[.,!&\'"-/]', '', name)
        
        # Replace multiple spaces with a single space and strip leading/trailing whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

