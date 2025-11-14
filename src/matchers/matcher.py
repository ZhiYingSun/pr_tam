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
    
    def __init__(self, incorporation_searcher):
        """
        Initialize restaurant matcher.
        
        Args:
            incorporation_searcher: IncorporationSearcher instance
        """
        self.incorporation_searcher = incorporation_searcher

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

    async def find_best_match(self, restaurant: RestaurantRecord) -> List[MatchResult]:
        """
        Searches for the top 25 matching business records for a given restaurant.
        Returns matches sorted by confidence score (highest first).
        Rate limiting handled by ZyteClient.
        """
        search_query = self._normalize_name(restaurant.name)
        candidates = await self.incorporation_searcher.search_business(
            search_query,
            limit=MatchingConfig.MAX_CANDIDATES
        )
        
        all_matches: List[MatchResult] = []
        
        # Calculate scores for all candidates
        for candidate_business in candidates:
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
        
        # Sort by confidence score (highest first) and take top 25
        all_matches.sort(key=lambda m: m.confidence_score, reverse=True)
        top_matches = all_matches[:25]
        
        if top_matches:
            accepted_count = sum(1 for m in top_matches if m.is_accepted)
            logger.info(
                f"Found {len(top_matches)} candidate(s) for '{restaurant.name}' "
                f"(top score: {top_matches[0].confidence_score:.1f}%, "
                f"{accepted_count} above threshold)"
            )
            return top_matches
        else:
            logger.info(f"No candidates found for '{restaurant.name}'")
            # Return a single MatchResult indicating no matches found
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

    async def match_multiple_restaurants(self, restaurants: List[RestaurantRecord]) -> List[MatchResult]:
        """
        Processes a list of restaurant records concurrently and attempts to find matches for each.
        Returns a flattened list of all match results (up to 25 per restaurant).
        Rate limiting handled by ZyteClient.
        """
        logger.info(f"Starting matching for {len(restaurants)} restaurants")
        
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
            elif isinstance(result, list):
                # Result is now a list of MatchResult objects (up to 25)
                matched_results.extend(result)
            elif result is not None:
                # Fallback for any other non-None result
                matched_results.append(result)
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during matching")
        
        logger.info(f"Matching completed: {len(matched_results)} total matches found from {len(restaurants)} restaurants")
        return matched_results
