"""
Data models for Puerto Rico Restaurant Matcher
"""
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class BusinessRecord:
    """Represents a business entity from Puerto Rico incorporation documents"""
    legal_name: str
    registration_number: str
    registration_index: str
    business_address: str
    status: str
    resident_agent_name: str
    resident_agent_address: str



@dataclass
class RestaurantRecord:
    """Represents a restaurant from Google Maps data"""
    name: str
    address: str
    city: str
    postal_code: str
    coordinates: Tuple[float, float]
    rating: float

    google_id: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    reviews_count: Optional[int] = None
    main_type: Optional[str] = None


@dataclass
class MatchResult:
    """Represents the result of matching a restaurant with a business record"""
    restaurant: RestaurantRecord
    business: BusinessRecord
    confidence_score: float
    match_type: str
    is_accepted: bool

    name_score: Optional[float] = None
    postal_code_match: Optional[bool] = None
    city_match: Optional[bool] = None
    match_reason: Optional[str] = None


class MatchingConfig:
    """Configuration constants for the matching algorithm"""
    
    # Matching thresholds
    NAME_MATCH_THRESHOLD = 70
    HIGH_CONFIDENCE_THRESHOLD = 85
    MEDIUM_CONFIDENCE_THRESHOLD = 75
    LOW_CONFIDENCE_THRESHOLD = 70
    
    # Score bonuses
    POSTAL_CODE_BONUS = 30
    CITY_MATCH_BONUS = 20
    
    # Search parameters
    MAX_CANDIDATES = 5
    REQUEST_DELAY = 0.5  # seconds between API calls
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    
    # Name normalization
    COMMON_SUFFIXES = [
        "llc", "inc", "corp", "ltd", "co", "restaurant", "bar", "cafe", "grill",
        "eats", "kitchen", "pub", "diner", "bistro", "pizzeria", "cantina",
        "taqueria", "bakery", "store", "market", "shop", "supercenter", "supermarket"
    ]
    PUNCTUATION_TO_REMOVE = r'[.,!&\'"-/]'
    
    # Output settings
    MIN_REVIEWS_FOR_MATCH = 5  # Minimum reviews to consider a restaurant
    MIN_RATING_FOR_MATCH = 3.0  # Minimum rating to consider a restaurant


def create_restaurant_from_csv_row(row: dict) -> RestaurantRecord:
    """Convert a CSV row to a RestaurantRecord"""
    # Handle NaN values by converting to empty strings
    def safe_str(value):
        if pd.isna(value) or value is None:
            return ''
        return str(value)
    
    return RestaurantRecord(
        name=safe_str(row.get('Name', '')),
        address=safe_str(row.get('Full address', '')),
        city=safe_str(row.get('City', '')),
        postal_code=safe_str(row.get('Postal code', '')),
        coordinates=(float(row.get('Longitude', 0)), float(row.get('Latitude', 0))),
        rating=float(row.get('Reviews rating', 0)),
        google_id=safe_str(row.get('Google ID', '')),
        phone=safe_str(row.get('Phone', '')),
        website=safe_str(row.get('Website', '')),
        reviews_count=int(row.get('Reviews count', 0)) if pd.notna(row.get('Reviews count')) else None,
        main_type=safe_str(row.get('Main type', ''))
    )


def create_business_from_api_response(response_data: dict) -> BusinessRecord:
    """Convert API response data to a BusinessRecord"""
    # Handle nested response structure
    actual_response = response_data.get('response', response_data)
    
    corporation = actual_response.get('corporation', {})
    corp_street_address = actual_response.get('corpStreetAddress', {})
    resident_agent = actual_response.get('residentAgent', {})
    
    # Extract legal name
    legal_name = corporation.get('corpName', '')
    
    # Extract registration number
    registration_number = corporation.get('corpRegisterNumber', '')
    
    # Extract street address
    street_address_parts = []
    if corp_street_address.get('address1'):
        street_address_parts.append(corp_street_address['address1'])
    if corp_street_address.get('address2'):
        street_address_parts.append(corp_street_address['address2'])
    
    # Build full address string
    address_parts = []
    if street_address_parts:
        address_parts.append(', '.join(street_address_parts))
    if corp_street_address.get('city'):
        address_parts.append(corp_street_address['city'])
    if corp_street_address.get('zip'):
        address_parts.append(corp_street_address['zip'])
    
    business_address = ', '.join(address_parts)
    
    # Extract status
    status = corporation.get('statusEn', '')
    
    # Extract resident agent name
    resident_agent_name = ''
    if resident_agent.get('individualName'):
        individual_name = resident_agent['individualName']
        name_parts = []
        if individual_name.get('firstName'):
            name_parts.append(individual_name['firstName'])
        if individual_name.get('middleName'):
            name_parts.append(individual_name['middleName'])
        if individual_name.get('lastName'):
            name_parts.append(individual_name['lastName'])
        if individual_name.get('surName'):
            name_parts.append(individual_name['surName'])
        resident_agent_name = ' '.join(name_parts)
    
    # Extract resident agent address
    resident_agent_address = ''
    if resident_agent.get('streetAddress'):
        agent_addr = resident_agent['streetAddress']
        agent_addr_parts = []
        if agent_addr.get('address1'):
            agent_addr_parts.append(agent_addr['address1'])
        if agent_addr.get('address2'):
            agent_addr_parts.append(agent_addr['address2'])
        resident_agent_address = ', '.join(agent_addr_parts)
    
    return BusinessRecord(
        legal_name=legal_name,
        registration_number=str(registration_number),
        business_address=business_address,
        status=status,
        resident_agent_name=resident_agent_name,
        resident_agent_address=resident_agent_address
    )


def determine_match_type(confidence_score: float) -> str:
    """Determine match type based on confidence score"""
    if confidence_score >= MatchingConfig.HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif confidence_score >= MatchingConfig.MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def is_match_accepted(confidence_score: float) -> bool:
    """Determine if a match should be accepted based on confidence score"""
    return confidence_score >= MatchingConfig.NAME_MATCH_THRESHOLD
