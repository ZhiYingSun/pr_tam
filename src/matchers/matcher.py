"""
Restaurant matching algorithm for Puerto Rico Restaurant Matcher
"""
import logging
from typing import List, Optional
from rapidfuzz import fuzz
from src.data.models import RestaurantRecord, BusinessRecord, MatchResult, MatchingConfig, determine_match_type, is_match_accepted

logger = logging.getLogger(__name__)


class RestaurantMatcher:
    """Core matching logic for matching restaurants with Puerto Rico incorporation documents"""
    
    def __init__(self, searcher, config: MatchingConfig = None):
        self.searcher = searcher
        self.config = config or MatchingConfig()
    
    def find_best_match(self, restaurant: RestaurantRecord) -> Optional[MatchResult]:
        """
        Find the best match for a restaurant in Puerto Rico incorporation documents.
        
        Args:
            restaurant: RestaurantRecord to match
            
        Returns:
            MatchResult if a good match is found, None otherwise
        """
        try:
            # Search for the restaurant name in PR incorporation docs
            candidates = self.searcher.search_business(restaurant.name, limit=self.config.MAX_CANDIDATES)
            
            if not candidates:
                logger.debug(f"No candidates found for '{restaurant.name}'")
                return None
            
            # Score each candidate
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                score = self._calculate_match_score(restaurant, candidate)
                
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            # Check if the best match meets our threshold
            if best_score >= self.config.NAME_MATCH_THRESHOLD:
                match_type = determine_match_type(best_score)
                is_accepted = is_match_accepted(best_score)
                
                return MatchResult(
                    restaurant=restaurant,
                    business=best_match,
                    confidence_score=best_score,
                    match_type=match_type,
                    is_accepted=is_accepted,
                    name_score=best_score,
                    postal_code_match=(restaurant.postal_code == self._extract_postal_code(best_match.business_address)),
                    city_match=(restaurant.city.lower() == self._extract_city(best_match.business_address).lower()),
                    match_reason=f"Name match: {best_score:.1f}%"
                )
            
            logger.debug(f"Best match for '{restaurant.name}' scored {best_score:.1f}%, below threshold {self.config.NAME_MATCH_THRESHOLD}")
            return None
            
        except Exception as e:
            logger.error(f"Error matching restaurant '{restaurant.name}': {e}")
            return None
    
    def _calculate_match_score(self, restaurant: RestaurantRecord, business: BusinessRecord) -> float:
        """
        Calculate match score between a restaurant and business record.
        Name match contributes 50% of the total score, location bonuses contribute the other 50%.
        
        Args:
            restaurant: RestaurantRecord from Google Maps
            business: BusinessRecord from PR incorporation docs
            
        Returns:
            Match score (0-100)
        """
        # Base score: name similarity (weighted to 50% of total)
        name_score = self._calculate_name_similarity(restaurant.name, business.legal_name)
        name_weighted = name_score * 0.5  # 50% weight (0-50 points)
        
        # Calculate location bonuses (contribute up to 50% of total)
        bonus_total = 0.0
        business_postal_code = self._extract_postal_code(business.business_address)
        business_city = self._extract_city(business.business_address)
        
        if restaurant.postal_code and business_postal_code and restaurant.postal_code == business_postal_code:
            bonus_total += self.config.POSTAL_CODE_BONUS
            logger.debug(f"Postal code match bonus: +{self.config.POSTAL_CODE_BONUS}")
        
        if restaurant.city and business_city and restaurant.city.lower() == business_city.lower():
            bonus_total += self.config.CITY_MATCH_BONUS
            logger.debug(f"City match bonus: +{self.config.CITY_MATCH_BONUS}")
        
        # Total score: name (0-50) + bonuses (0-50) = 0-100
        total_score = name_weighted + bonus_total
        
        # Cap the score at 100
        return min(total_score, 100.0)
    
    def _calculate_name_similarity(self, restaurant_name: str, business_name: str) -> float:
        """
        Calculate name similarity using RapidFuzz.
        
        Args:
            restaurant_name: Name from Google Maps
            business_name: Legal name from PR incorporation docs
            
        Returns:
            Similarity score (0-100)
        """
        # Normalize both names
        norm_restaurant = self._normalize_name(restaurant_name)
        norm_business = self._normalize_name(business_name)
        
        # Use token_sort_ratio for better matching of reordered words
        similarity = fuzz.token_sort_ratio(norm_restaurant, norm_business)
        
        logger.debug(f"Name similarity: '{norm_restaurant}' vs '{norm_business}' = {similarity:.1f}%")
        
        return similarity
    
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
        
        # Split by comma and take the second-to-last part (usually the city)
        parts = [part.strip() for part in address.split(',')]
        if len(parts) >= 2:
            # Usually city is the second-to-last part before postal code
            return parts[-2] if len(parts) > 2 else parts[-1]
        
        return ''
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize business name for better matching.
        
        Args:
            name: Business name to normalize
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common punctuation
        punctuation_to_remove = [',', '.', '!', '?', ':', ';', '&', "'", '"']
        for punct in punctuation_to_remove:
            normalized = normalized.replace(punct, '')
        
        # Remove common business suffixes
        business_suffixes = [
            'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 
            'co', 'company', 'restaurant', 'rest', 'bar', 'grill',
            'cafe', 'coffee', 'shop', 'store', 'food', 'kitchen'
        ]
        
        for suffix in business_suffixes:
            # Remove suffix if it's at the end of the name
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)]
            elif normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def match_multiple_restaurants(self, restaurants: List[RestaurantRecord]) -> List[MatchResult]:
        """
        Match multiple restaurants and return all results.
        
        Args:
            restaurants: List of RestaurantRecord objects
            
        Returns:
            List of MatchResult objects (including unmatched restaurants)
        """
        results = []
        
        for i, restaurant in enumerate(restaurants):
            logger.info(f"Processing restaurant {i+1}/{len(restaurants)}: {restaurant.name}")
            
            match = self.find_best_match(restaurant)
            if match:
                results.append(match)
            else:
                # Create a "no match" result for tracking
                no_match = MatchResult(
                    restaurant=restaurant,
                    business=None,
                    confidence_score=0.0,
                    match_type="none",
                    is_accepted=False,
                    match_reason="No suitable match found"
                )
                results.append(no_match)
        
        return results
