"""
Async Puerto Rico incorporation documents API client with connection pooling
"""
import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional

from src.data.models import (
    BusinessRecord,
    CorporationSearchResponse,
    CorporationDetailResponse,
    CorporationDetailResponseData,
    CorporationSearchRecord,
)
from src.searchers.zyte_client import ZyteClient
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class AsyncIncorporationSearcher:

    def __init__(self, zyte_api_key: str, max_concurrent: int = 20):
        self.zyte_api_key = zyte_api_key
        self.max_concurrent = max_concurrent
        self.zyte_client = ZyteClient(zyte_api_key)
        self.search_url = "https://rceapi.estado.pr.gov/api/corporation/search"
        
    async def __aenter__(self):
        """Async context manager entry - initialize ZyteClient."""
        await self.zyte_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close ZyteClient session."""
        await self.zyte_client.close()
    
    async def search_business_async(self, business_name: str, limit: int = 5) -> List[BusinessRecord]:
        """Search for business by name using reverse engineered PR incorporation API."""
        
        payload = {
            "cancellationMode": False,
            "comparisonType": 1,
            "corpName": business_name,
            "isWorkFlowSearch": False,
            "limit": limit,
            "matchType": 4,
            "onlyActive": True
        }
        
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/problem+json; charset=UTF-8',
            'Origin': 'https://rcp.estado.pr.gov',
            'Authorization': 'null'
        }
        
        try:
            # Use Zyte for rate-limited requests
            response = await self._make_corporation_search_request_async(payload, headers)
            logger.debug(f"Search response for '{business_name}': {response}")
            
            if response and response.response and response.response.records:
                records = response.response.records
                logger.info(f"Found {len(records)} records for '{business_name}'")
                business_records = []
                
                # Get detailed information for each business
                for record in records:
                    try:
                        business_entity_id = record.businessEntityId
                        logger.debug(f"Search result for '{business_name}': corpName='{record.corpName}', businessEntityId={business_entity_id}")
                        
                        if business_entity_id:
                            # Get detailed business information using registrationIndex
                            registration_index = record.registrationIndex
                            detailed_record = await self._get_business_details_async(business_entity_id, registration_index)
                            if detailed_record:
                                business_record = self._create_business_record_from_details(detailed_record)
                                logger.debug(f"Successfully created detailed record for {business_entity_id}")
                            else:
                                # Fallback to search result if details fail
                                business_record = self._create_business_record_from_search(record)
                                logger.debug(f"Fell back to search result for {business_entity_id}")
                        else:
                            # Fallback to search result if no businessEntityId
                            business_record = self._create_business_record_from_search(record)
                            logger.debug(f"No businessEntityId, using search result")
                        
                        business_records.append(business_record)
                        logger.debug(f"Created business record: {business_record.legal_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create business record: {e}")
                        continue
                
                return business_records
            else:
                logger.warning(f"No records found for '{business_name}'")
                return []
            
        except Exception as e:
            logger.error(f"Error searching for business '{business_name}': {e}")
            return []
    
    async def _make_corporation_search_request_async(self, payload: Dict, headers: Dict) -> Optional[CorporationSearchResponse]:
        try:
            logger.debug(f"Making async request through Zyte")

            zyte_response = await self.zyte_client.post_request(
                url=self.search_url,
                request_body=payload,
                headers=headers
            )

            try:
                decoded_body = zyte_response.decode_body()
                search_response = CorporationSearchResponse(**decoded_body)
                return search_response
            except ValidationError as e:
                logger.error(f"Failed to parse PR search response: {e}")
                return None
                    
        except Exception as e:
            logger.error(f"Zyte async POST request failed: {e}")
            return None
    
    async def _get_business_details_async(self, business_entity_id: int, registration_index: str = None) -> Optional[CorporationDetailResponseData]:
        try:
            # Try different URL patterns based on what we have available
            url = f"https://rceapi.estado.pr.gov/api/corporation/info/{registration_index}"

            
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Origin': 'https://rcp.estado.pr.gov',
                'Authorization': 'null'
            }
            

            logger.debug(f"Trying detail URL with registrationIndex: {url}")
            response = await self._make_corporation_detail_get_request_async(url, headers)
            logger.debug(f"Detail response type: {type(response)}, value: {response}")
                
            if response and response.response and response.response.corporation:
                logger.debug(f"Successfully retrieved details for business entity {business_entity_id} using URL: {url}")
                return response.response
            else:
                logger.debug(f"URL {url} failed, trying next...")

            return None
                
        except Exception as e:
            logger.error(f"Error getting business details for entity {business_entity_id}: {e}")
            return None
    
    async def _make_corporation_detail_get_request_async(self, url: str, headers: Dict) -> Optional[CorporationDetailResponse]:
        try:
            logger.debug(f"Making async GET request through Zyte to {url}")

            zyte_response = await self.zyte_client.get_request(
                url=url,
                headers=headers
            )

            try:
                decoded_body = zyte_response.decode_body()
                corporation_detail = CorporationDetailResponse(**decoded_body)
                return corporation_detail
            except ValidationError as e:
                logger.error(f"Failed to parse PR detail response: {e}")
                return None
                    
        except Exception as e:
            logger.error(f"Zyte async GET request failed: {e}")
            return None
    
    def _create_business_record_from_details(self, details: CorporationDetailResponseData) -> BusinessRecord:
        corporation = details.corporation
        main_location = details.mainLocation
        resident_agent = details.residentAgent
        
        # Extract address information
        business_address = ''
        
        if main_location and main_location.streetAddress:
            street_addr = main_location.streetAddress
            address_parts = []
            if street_addr.address1:
                address_parts.append(street_addr.address1)
            if street_addr.address2:
                address_parts.append(street_addr.address2)
            if street_addr.city:
                address_parts.append(street_addr.city)
            if street_addr.zip:
                address_parts.append(street_addr.zip)
            business_address = ', '.join(address_parts)
        
        # Extract resident agent information
        resident_agent_name = ''
        resident_agent_address = ''
        
        if resident_agent:
            if resident_agent.isIndividual and resident_agent.individualName:
                individual = resident_agent.individualName
                name_parts = []
                if individual.firstName:
                    name_parts.append(individual.firstName)
                if individual.middleName:
                    name_parts.append(individual.middleName)
                if individual.lastName:
                    name_parts.append(individual.lastName)
                if individual.surName:
                    name_parts.append(individual.surName)
                resident_agent_name = ' '.join(name_parts)
            elif resident_agent.organizationName:
                resident_agent_name = resident_agent.organizationName.name or ''
            
            if resident_agent.streetAddress:
                agent_addr = resident_agent.streetAddress
                address_parts = []
                if agent_addr.address1:
                    address_parts.append(agent_addr.address1)
                if agent_addr.address2:
                    address_parts.append(agent_addr.address2)
                resident_agent_address = ' '.join(address_parts).strip()

        return BusinessRecord.from_corporation(
            corporation,
            business_address=business_address,
            resident_agent_name=resident_agent_name,
            resident_agent_address=resident_agent_address,
        )
    
    def _create_business_record_from_search(self, record: CorporationSearchRecord) -> BusinessRecord:
        return BusinessRecord(
            legal_name=record.corpName or '',
            registration_number=str(record.registrationNumber) if record.registrationNumber else '',
            registration_index=record.registrationIndex or '',
            status=record.statusEn or '',
        )


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
        """Normalize name for matching (same logic as AsyncRestaurantMatcher)"""
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
