"""
Restaurant Matcher
"""
import re
import logging
import asyncio
from typing import List, Optional, Tuple
from rapidfuzz import fuzz
from src.models.models import RestaurantRecord, BusinessRecord, MatchResult, MatchingConfig, determine_match_type

logger = logging.getLogger(__name__)


class RestaurantMatcher:
    """
    Matcher for restaurant records.
    """
    
    def __init__(self, incorporation_searcher, openai_client):
        """
        Initialize restaurant matcher.
        
        Args:
            incorporation_searcher: IncorporationSearcher instance
            openai_client: Optional OpenAIClient instance for name cleaning
        """
        self.incorporation_searcher = incorporation_searcher
        self.openai_client = openai_client

    def _normalize_name(self, name: str) -> str:
        """
        Normalizes a business name for better matching.
        - Converts to lowercase
        - Removes common business suffixes (LLC, Inc, Corp, Restaurant, etc.)
        - Removes punctuation
        - Strips extra whitespace
        """
        if not name or not isinstance(name, str):
            return ""
            
        name = name.lower()
        
        # Remove common suffixes
        for suffix in MatchingConfig.COMMON_SUFFIXES:
            name = re.sub(r'\b' + re.escape(suffix) + r'\b', '', name)
        
        # Remove punctuation
        name = re.sub(MatchingConfig.PUNCTUATION_TO_REMOVE, '', name)
        
        # Replace multiple spaces with a single space and strip leading/trailing whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    async def _clean_name_with_openai(self, restaurant_name: str) -> str:
        """
        Uses OpenAI to extract the core business name, removing unimportant words
        that might interfere with the business registry search.
        
        Args:
            restaurant_name: The restaurant name from Google Maps
            
        Returns:
            Cleaned business name suitable for searching business registries
        """
        normalized_name = self._normalize_name(restaurant_name)
        
        try:
            prompt = f"""Extract the core business name from this restaurant name for searching a business registry.

Restaurant name: "{normalized_name}"

Remove:
- Location descriptors (e.g., "San Juan", "Miramar", "Plaza")
- Generic business types (e.g., "Restaurant", "Cafe", "Bar", "Grill")
- Descriptive words that are not part of the legal entity name
- Articles (a, the, etc.)

Keep:
- The distinctive brand/business name
- Any words that are likely part of the legal entity name

Return ONLY the cleaned name, nothing else. If uncertain, return the original name.

Examples:
- "Pizza e Birra Miramar" → "Pizza e Birra"
- "The Yellow Door, Coffee & Ice Cream Shop" → "Yellow Door Coffee Ice Cream Shop"
- "Ocean Lab Brewing Co." → "Ocean Lab Brewing"
- "Lote 23" → "Lote 23"

Cleaned name:"""

            response = await self.openai_client.chat_completion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts core business names for registry searches."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            if response and "content" in response:
                cleaned_name = response["content"].strip().strip('"').strip("'")
                if cleaned_name:
                    logger.info(f"OpenAI cleaned name: '{restaurant_name}' → '{cleaned_name}'")
                    return cleaned_name
            
            logger.warning(f"OpenAI returned empty response for '{restaurant_name}', using standard normalization")
            return self._normalize_name(restaurant_name)
            
        except Exception as e:
            logger.error(f"Error cleaning name with OpenAI: {e}. Falling back to standard normalization")
            return self._normalize_name(restaurant_name)

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculates the similarity between two normalized names using RapidFuzz's token_sort_ratio.
        """
        normalized_name1 = self._normalize_name(name1)
        normalized_name2 = self._normalize_name(name2)

        score = fuzz.token_sort_ratio(normalized_name1, normalized_name2)
        return score

    def _calculate_match_score(
        self,
        restaurant: RestaurantRecord,
        business: BusinessRecord,
        name_score: float
    ) -> Tuple[float, str, bool, bool]:
        """
        Calculates a match score based on name similarity and location data.
        """
        name_weighted = name_score * 0.5
        match_reason_parts = [f"Name match: {name_score:.1f}% (weighted: {name_weighted:.1f}%)"]

        bonus_total = 0.0
        postal_code_match = False
        if restaurant.postal_code and business.business_address:
            business_postal_code = self._extract_postal_code(business.business_address)
            if business_postal_code and restaurant.postal_code == business_postal_code:
                bonus_total += MatchingConfig.POSTAL_CODE_BONUS
                postal_code_match = True
                match_reason_parts.append(f"Postal code bonus: +{MatchingConfig.POSTAL_CODE_BONUS}")
        
        city_match = False
        if restaurant.city and business.business_address:
            business_city = self._extract_city(business.business_address)
            if business_city and restaurant.city.lower() == business_city.lower():
                bonus_total += MatchingConfig.CITY_MATCH_BONUS
                city_match = True
                match_reason_parts.append(f"City bonus: +{MatchingConfig.CITY_MATCH_BONUS}")

        final_score = name_weighted + bonus_total
        
        return final_score, "; ".join(match_reason_parts), postal_code_match, city_match
    
    def _extract_postal_code(self, address: str) -> str:
        """Extract postal code from business address string"""
        if not address:
            return ''
        
        # Look for 5-digit postal code pattern
        import re
        postal_code_match = re.search(r'\b(\d{5})\b', address)
        if postal_code_match:
            return postal_code_match.group(1)
        
        return ''
    
    def _extract_city(self, address: str) -> str:
        """Extract city from business address string"""
        if not address:
            return ''

        # TODO fix using better heuristic later
        # Split by comma and take the second-to-last part (usually the city)
        parts = [part.strip() for part in address.split(',')]
        if len(parts) >= 2:
            # Usually city is the second-to-last part before postal code
            return parts[-2] if len(parts) > 2 else parts[-1]
        
        return ''

    async def find_best_match(self, restaurant: RestaurantRecord) -> List[MatchResult]:
        """
        Searches for the top 25 matching business records for a given restaurant.
        Uses OpenAI to clean the name before searching.
        """
        search_query = await self._clean_name_with_openai(restaurant.name)
        search_records = await self.incorporation_searcher.search_business(
            search_query,
            limit=MatchingConfig.SEARCH_LIMIT
        )
        
        if not search_records:
            logger.info(f"No candidates found for '{restaurant.name}'")
            return [MatchResult(
                restaurant=restaurant,
                business=None,
                confidence_score=0.0,
                match_type="none",
                is_accepted=False,
                name_score=0.0,
                postal_code_match=False,
                city_match=False,
                match_reason="No candidates found"
            )]
        
        # Step 1: Do fuzzy matching on corpName from lightweight search records
        scored_records = []
        for record in search_records:
            if record.corpName:
                name_score = self._calculate_name_similarity(restaurant.name, record.corpName)
                scored_records.append((name_score, record))
        
        # Sort by fuzzy score and take top 3
        scored_records.sort(key=lambda x: x[0], reverse=True)
        top_3_records = [record for _, record in scored_records[:3]]
        
        logger.info(f"Fuzzy matched {len(search_records)} records, fetching details for top {len(top_3_records)}")
        
        # Step 2: Only fetch detailed info for top 3 fuzzy matches
        detailed_businesses = await self.incorporation_searcher.get_business_details_for_records(top_3_records)
        
        # Step 3: Calculate final scores with detailed information
        all_matches: List[MatchResult] = []
        
        for candidate_business in detailed_businesses:
            name_score = self._calculate_name_similarity(restaurant.name, candidate_business.legal_name)
            
            current_score, match_reason, postal_code_match, city_match = \
                self._calculate_match_score(restaurant, candidate_business, name_score)
            
            match_type = determine_match_type(current_score)
            is_accepted = current_score >= MatchingConfig.NAME_MATCH_THRESHOLD
            
            match_result = MatchResult(
                restaurant=restaurant,
                business=candidate_business,
                confidence_score=current_score,
                match_type=match_type,
                is_accepted=is_accepted,
                name_score=name_score,
                postal_code_match=postal_code_match,
                city_match=city_match,
                match_reason=match_reason
            )
            
            all_matches.append(match_result)

        all_matches.sort(key=lambda m: m.confidence_score, reverse=True)
        
        if all_matches:
            return all_matches
        else:
            logger.info(f"No valid matches found for '{restaurant.name}'")
            return [MatchResult(
                restaurant=restaurant,
                business=None,
                confidence_score=0.0,
                match_type="none",
                is_accepted=False,
                name_score=0.0,
                postal_code_match=False,
                city_match=False,
                match_reason="No valid matches found"
            )]
