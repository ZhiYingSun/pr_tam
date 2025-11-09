"""
Restaurant Matcher
"""
import re
import logging
import asyncio
from typing import List, Optional, Tuple
from rapidfuzz import fuzz
from src.data.models import RestaurantRecord, BusinessRecord, MatchResult, MatchingConfig, determine_match_type

logger = logging.getLogger(__name__)


class RestaurantMatcher:
    """
    Matcher for restaurant records.
    """
    
    def __init__(self, incorporation_searcher, max_concurrent: int = 20):
        """
        Initialize restaurant matcher.
        
        Args:
            incorporation_searcher: IncorporationSearcher instance
            max_concurrent: Deprecated - kept for compatibility but not used.
                           Rate limiting is handled by ZyteClient.
        """
        self.incorporation_searcher = incorporation_searcher
        self.max_concurrent = max_concurrent  # Kept for compatibility

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
        Calculates a comprehensive match score based on name similarity and location data.
        Name match contributes 50% of the total score, location bonuses contribute the other 50%.
        Returns the total score, match reason, postal code match status, and city match status.
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

        final_score = min(name_weighted + bonus_total, 100.0)
        
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

    async def find_best_match(self, restaurant: RestaurantRecord) -> Optional[MatchResult]:
        """
        Searches for the best matching business record for a given restaurant.
        Rate limiting handled by ZyteClient.
        """
        search_query = self._normalize_name(restaurant.name)
        candidates = await self.incorporation_searcher.search_business(
            search_query,
            limit=MatchingConfig.MAX_CANDIDATES
        )
        
        best_match_result: Optional[MatchResult] = None
        best_score = 0.0
        
        for candidate_business in candidates:
            name_score = self._calculate_name_similarity(restaurant.name, candidate_business.legal_name)
            
            current_score, match_reason, postal_code_match, city_match = \
                self._calculate_match_score(restaurant, candidate_business, name_score)
            
            if current_score > best_score:
                best_score = current_score
                match_type = determine_match_type(best_score)
                is_accepted = best_score >= MatchingConfig.NAME_MATCH_THRESHOLD
                
                best_match_result = MatchResult(
                    restaurant=restaurant,
                    business=candidate_business,
                    confidence_score=best_score,
                    match_type=match_type,
                    is_accepted=is_accepted,
                    name_score=name_score,
                    postal_code_match=postal_code_match,
                    city_match=city_match,
                    match_reason=match_reason
                )
        
        if best_match_result and best_match_result.is_accepted:
            logger.info(
                f"Matched '{restaurant.name}' with '{best_match_result.business.legal_name}' "
                f"Confidence: {best_match_result.confidence_score:.1f}% ({best_match_result.match_type})"
            )
            return best_match_result
        
        if best_match_result:
            logger.info(
                f"No accepted match for '{restaurant.name}'. Best candidate: "
                f"'{best_match_result.business.legal_name}' with {best_match_result.confidence_score:.1f}% ({best_match_result.match_type})"
            )
            return best_match_result
        else:
            logger.info(f"No candidates found for '{restaurant.name}'")
            # Create a MatchResult for unmatched restaurants
            return MatchResult(
                restaurant=restaurant,
                business=None,
                confidence_score=0.0,
                match_type="none",
                is_accepted=False,
                name_score=0.0,
                postal_code_match=False,
                city_match=False,
                match_reason="No candidates found"
            )

    async def match_multiple_restaurants(self, restaurants: List[RestaurantRecord]) -> List[MatchResult]:
        """
        Processes a list of restaurant records concurrently and attempts to find matches for each.
        Rate limiting handled by ZyteClient.
        """
        logger.info(f"Starting matching for {len(restaurants)} restaurants with max_concurrent={self.max_concurrent}")
        
        # Create tasks for all restaurants
        tasks = [self.find_best_match(restaurant) for restaurant in restaurants]
        
        # Execute all tasks concurrently with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful matches and log errors
        matched_results: List[MatchResult] = []
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error matching restaurant '{restaurants[i].name}': {result}")
                error_count += 1
            elif result is not None:
                matched_results.append(result)
            # Note: result can be None if there was an error, but we now always return MatchResult
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during matching")
        
        logger.info(f"Matching completed: {len(matched_results)} matches found from {len(restaurants)} restaurants")
        return matched_results
