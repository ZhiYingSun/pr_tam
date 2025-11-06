"""
Async Puerto Rico incorporation documents API client with connection pooling
"""
import json
import logging
import asyncio
import re
import base64
from typing import Dict, List, Any, Optional
import aiohttp
from aiohttp import ClientSession, ClientTimeout, BasicAuth

from src.data.models import BusinessRecord

logger = logging.getLogger(__name__)


class AsyncIncorporationSearcher:

    def __init__(self, zyte_api_key: str, max_concurrent: int = 20):
        self.zyte_api_key = zyte_api_key
        self.max_concurrent = max_concurrent
        self.session = None

        self.search_url = "https://rceapi.estado.pr.gov/api/corporation/search"
        
    async def __aenter__(self):
        """Async context manager entry - create session with connection pooling."""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        timeout = ClientTimeout(total=30, connect=10)
        self.session = ClientSession(
            connector=connector, 
            timeout=timeout,
            auth=BasicAuth(self.zyte_api_key, "")
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session."""
        if self.session:
            await self.session.close()
    
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
            response = await self._make_zyte_request_async(payload, headers)
            logger.debug(f"Search response for '{business_name}': {response}")
            
            if response and 'response' in response and 'records' in response['response']:
                records = response['response']['records']
                logger.info(f"Found {len(records)} records for '{business_name}'")
                business_records = []
                
                # Get detailed information for each business
                for record in records:
                    try:
                        business_entity_id = record.get('businessEntityId')
                        logger.debug(f"Search result for '{business_name}': corpName='{record.get('corpName')}', businessEntityId={business_entity_id}")
                        
                        if business_entity_id:
                            # Get detailed business information using registrationIndex
                            registration_index = record.get('registrationIndex')
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
    
    async def _make_zyte_request_async(self, payload: Dict, headers: Dict) -> Optional[Dict]:
        if not self.session:
            logger.error("Session not initialized. Cannot make request.")
            return None
        
        try:
            logger.debug(f"Making async request through Zyte")
            
            # Prepare Zyte payload
            zyte_payload = {
                "url": self.search_url,
                "httpResponseBody": True,
                "httpRequestMethod": "POST",
                "httpRequestText": json.dumps(payload),
                "customHttpRequestHeaders": [
                    {"name": k, "value": v} for k, v in headers.items()
                ]
            }
            
            async with self.session.post(
                "https://api.zyte.com/v1/extract",
                json=zyte_payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Zyte API returned status {resp.status}")
                    return None
                    
                try:
                    data = await resp.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse Zyte response as JSON: {json_error}")
                    return None
                
                if data.get('httpResponseBody', None):
                    try:
                        # Decode base64 response body
                        decoded_body = base64.b64decode(data['httpResponseBody']).decode('utf-8')
                        body = json.loads(decoded_body)
                        return body
                    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as decode_error:
                        logger.error(f"Failed to decode/parse httpResponseBody: {decode_error}")
                        logger.debug(f"Response body: {data['httpResponseBody'][:200]}...")
                        return None
                else:
                    logger.warning(f"No httpResponseBody in Zyte response: {data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Zyte async POST request failed: {e}")
            return None
    
    async def _get_business_details_async(self, business_entity_id: int, registration_index: str = None) -> Optional[Dict]:
        try:
            # Try different URL patterns based on what we have available
            urls_to_try = []
            
            if registration_index:
                urls_to_try.append(f"https://rceapi.estado.pr.gov/api/corporation/info/{registration_index}")
            
            # Fallback to business entity ID
            urls_to_try.append(f"https://rceapi.estado.pr.gov/api/corporation/{business_entity_id}")
            
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Origin': 'https://rcp.estado.pr.gov',
                'Authorization': 'null'
            }
            
            # Try each URL until one works
            for detail_url in urls_to_try:
                logger.debug(f"Trying detail URL with registrationIndex: {detail_url}")
                response = await self._make_zyte_get_request_async(detail_url, headers)
                logger.debug(f"Detail response type: {type(response)}, value: {response}")
                
                if response and isinstance(response, dict) and 'response' in response and response['response'] and 'corporation' in response['response']:
                    logger.debug(f"Successfully retrieved details for business entity {business_entity_id} using URL: {detail_url}")
                    return response['response']
                else:
                    logger.debug(f"URL {detail_url} failed, trying next...")
            
            logger.warning(f"No detailed information found for business entity {business_entity_id} after trying {len(urls_to_try)} URLs")
            return None
                
        except Exception as e:
            logger.error(f"Error getting business details for entity {business_entity_id}: {e}")
            return None
    
    async def _make_zyte_get_request_async(self, url: str, headers: Dict) -> Optional[Dict]:
        if not self.session:
            logger.error("Session not initialized. Cannot make request.")
            return None
        
        try:
            logger.debug(f"Making async GET request through Zyte to {url}")
            
            # Prepare Zyte payload for GET request
            zyte_payload = {
                "url": url,
                "httpResponseBody": True,
                "httpRequestMethod": "GET",
                "customHttpRequestHeaders": [
                    {"name": k, "value": v} for k, v in headers.items()
                ]
            }
            
            async with self.session.post(
                "https://api.zyte.com/v1/extract",
                json=zyte_payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Zyte API returned status {resp.status}")
                    return None
                    
                try:
                    data = await resp.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse Zyte response as JSON: {json_error}")
                    return None
                
                if data and 'httpResponseBody' in data and data['httpResponseBody']:
                    try:
                        # Decode base64 response body
                        decoded_body = base64.b64decode(data['httpResponseBody']).decode('utf-8')
                        body = json.loads(decoded_body)
                        return body
                    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as decode_error:
                        logger.error(f"Failed to decode/parse httpResponseBody: {decode_error}")
                        logger.debug(f"Response body: {data['httpResponseBody'][:200]}...")
                        return None
                else:
                    logger.warning(f"No httpResponseBody in Zyte response: {data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Zyte async GET request failed: {e}")
            return None
    
    def _create_business_record_from_details(self, details: Dict) -> BusinessRecord:
        corporation = details.get('corporation', {})
        main_location = details.get('mainLocation', {})
        resident_agent = details.get('residentAgent', {})
        
        # Extract address information
        business_address = ''
        
        if main_location and 'streetAddress' in main_location:
            street_addr = main_location['streetAddress']
            address_parts = []
            if street_addr.get('address1'):
                address_parts.append(street_addr['address1'])
            if street_addr.get('address2'):
                address_parts.append(street_addr['address2'])
            if street_addr.get('city'):
                address_parts.append(street_addr['city'])
            if street_addr.get('zip'):
                address_parts.append(street_addr['zip'])
            business_address = ', '.join(address_parts)
        
        # Extract resident agent information
        resident_agent_name = ''
        resident_agent_address = ''
        
        if resident_agent:
            if resident_agent.get('isIndividual') and 'individualName' in resident_agent:
                individual = resident_agent['individualName']
                name_parts = []
                if individual.get('firstName'):
                    name_parts.append(individual['firstName'])
                if individual.get('middleName'):
                    name_parts.append(individual['middleName'])
                if individual.get('lastName'):
                    name_parts.append(individual['lastName'])
                if individual.get('surName'):
                    name_parts.append(individual['surName'])
                resident_agent_name = ' '.join(name_parts)
            elif 'organizationName' in resident_agent:
                resident_agent_name = resident_agent['organizationName'].get('name', '')
            
            if 'streetAddress' in resident_agent:
                agent_addr = resident_agent['streetAddress']
                resident_agent_address = f"{agent_addr.get('address1', '')} {agent_addr.get('address2', '')}".strip()
        
        return BusinessRecord(
            legal_name=corporation.get('corpName', ''),
            registration_number=str(corporation.get('corpRegisterNumber', '')),
            registration_index=corporation.get('corpRegisterIndex', ''),
            business_address=business_address,
            status=corporation.get('statusEn', ''),
            resident_agent_name=resident_agent_name,
            resident_agent_address=resident_agent_address
        )
    
    def _create_business_record_from_search(self, record: Dict) -> BusinessRecord:
        return BusinessRecord(
            legal_name=record.get('corpName', ''),
            registration_number=str(record.get('registrationNumber', '')),
            registration_index=record.get('registrationIndex', ''),
            business_address='',  # Not available in search results
            status=record.get('statusEn', ''),
            resident_agent_name='',  # Not available in search results
            resident_agent_address=''  # Not available in search results
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
                    street_address='',  # Mock data
                    city='',  # Mock data
                    postal_code='',  # Mock data
                    status=record.get('statusEn', ''),
                    resident_agent_name='',  # Mock data
                    resident_agent_address=''  # Mock data
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
