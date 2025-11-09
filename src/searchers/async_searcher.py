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
