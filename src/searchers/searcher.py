"""
Puerto Rico incorporation documents API client
"""
import logging
from typing import Dict, List, Any, Optional
from src.searchers.zyte_client import ZyteClient
from src.data.models import BusinessRecord, create_business_from_api_response

logger = logging.getLogger(__name__)


class IncorporationSearcher:
    """Client for Puerto Rico incorporation documents API."""
    
    def __init__(self, zyte_api_key: str, timeout: int = 30):
        self.zyte_api_key = zyte_api_key
        self.timeout = timeout
        self.zyte_client = ZyteClient(zyte_api_key) if zyte_api_key else None
        
        # Puerto Rico API endpoints
        self.search_url = "https://rceapi.estado.pr.gov/api/corporation/search"
        
    def search_business(self, business_name: str, limit: int = 5) -> List[BusinessRecord]:
        """Search for business by name using the Puerto Rico API."""
        
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
            response = self._make_zyte_request(payload, headers)
            logger.debug(f"Search response: {response}")
            
            if response and 'response' in response and 'records' in response['response']:
                records = response['response']['records']
                logger.info(f"Found {len(records)} records for '{business_name}'")
                business_records = []
                
                # Get detailed information for each business
                for record in records:
                    try:
                        business_entity_id = record.get('businessEntityId')
                        if business_entity_id:
                            # Get detailed business information using registrationIndex
                            registration_index = record.get('registrationIndex')
                            detailed_record = self._get_business_details(business_entity_id, registration_index)
                            if detailed_record:
                                business_record = self._create_business_record_from_details(detailed_record)
                            else:
                                # Fallback to search result if details fail
                                business_record = self._create_business_record_from_search(record)
                        else:
                            # Fallback to search result if no businessEntityId
                            business_record = self._create_business_record_from_search(record)
                        
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
    
    def get_business_details(self, registry_number: str) -> Optional[BusinessRecord]:
        """Get detailed information for a specific business using GET request."""
        
        # Construct the details URL using the registration number
        details_url = f"https://rceapi.estado.pr.gov/api/corporation/info/{registry_number}"
        
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://rcp.estado.pr.gov',
            'Authorization': 'null'
        }
        
        try:
            # Use Zyte GET request for rate limiting
            response_text = self._make_zyte_get_request(details_url, headers)
            if response_text:
                import json
                response_data = json.loads(response_text)
                logger.debug(f"Details response for {registry_number}: {response_data}")
                return create_business_from_api_response(response_data)
            else:
                logger.warning(f"No response text for registry number '{registry_number}'")
                return None
                
        except Exception as e:
            logger.error(f"Error getting details for registry number '{registry_number}': {e}")
            return None
    
    def _make_zyte_request(self, payload: Dict, headers: Dict) -> Optional[Dict]:
        """Make request through Zyte to avoid rate limiting."""
        try:
            logger.info("Making request through Zyte")
            response = self.zyte_client.post_request_sync(
                self.search_url,
                payload,
                headers
            )
            return response
        except Exception as e:
            logger.error(f"Zyte request failed: {e}")
            return None
    
    def _make_zyte_get_request(self, url: str, headers: Dict) -> Optional[str]:
        """Make GET request through Zyte to avoid rate limiting."""
        try:
            logger.info(f"Making GET request through Zyte to: {url}")
            response = self.zyte_client.get_request_sync(url, headers)
            return response
        except Exception as e:
            logger.error(f"Zyte GET request failed: {e}")
            return None
    
    def _get_business_details(self, business_entity_id: int, registration_index: str = None) -> Optional[Dict]:
        """Get detailed business information using GET request with registrationIndex."""
        try:
            # Try different URL patterns based on what we have available
            urls_to_try = []
            
            if registration_index:
                # Try using registration index first (e.g., "528170-1511")
                urls_to_try.append(f"https://rceapi.estado.pr.gov/api/corporation/{registration_index}")
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
                response_text = self._make_zyte_get_request(detail_url, headers)
                
                if response_text:
                    try:
                        import json
                        response = json.loads(response_text)
                        if response and 'response' in response and response['response'] and 'corporation' in response['response']:
                            logger.debug(f"Successfully retrieved details for business entity {business_entity_id} using URL: {detail_url}")
                            return response['response']
                        else:
                            logger.debug(f"URL {detail_url} failed, trying next...")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse response from {detail_url}: {e}")
                        continue
                else:
                    logger.debug(f"Empty response from {detail_url}, trying next...")
            
            logger.warning(f"No detailed information found for business entity {business_entity_id} after trying {len(urls_to_try)} URLs")
            return None
                
        except Exception as e:
            logger.error(f"Error getting business details for entity {business_entity_id}: {e}")
            return None
    
    def _create_business_record_from_details(self, details: Dict) -> BusinessRecord:
        """Create BusinessRecord from detailed API response."""
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
        """Create BusinessRecord from search result (fallback)."""
        return BusinessRecord(
            legal_name=record.get('corpName', ''),
            registration_number=str(record.get('registrationNumber', '')),
            registration_index=record.get('registrationIndex', ''),
            business_address='',  # Not available in search results
            status=record.get('statusEn', ''),
            resident_agent_name='',  # Not available in search results
            resident_agent_address=''  # Not available in search results
        )
