"""
Restaurant Matcher
"""
import re
import logging
import json
from typing import List, Tuple
from rapidfuzz import fuzz

from src.models.api_models import CorporationSearchRecord
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
        if not name or not isinstance(name, str):
            return ""

        normalized = name.lower()

        # Remove content in parentheses
        normalized = re.sub(r'\([^)]*\)', '', normalized)

        # Remove legal suffixes (case insensitive)
        normalized = re.sub(MatchingConfig.LEGAL_TERMS, '', normalized, flags=re.IGNORECASE)

        # Remove common words
        normalized = re.sub(MatchingConfig.COMMON_WORDS, '', normalized, flags=re.IGNORECASE)

        # Remove all punctuation except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def _clean_json_response(self, content:str):
        """Remove markdown code fences from JSON response"""
        json = content.strip()
        # Remove ```json or ``` at start and ``` at end
        json = re.sub(r'^```(?:json)?\s*', '', json)
        json = re.sub(r'\s*```$', '', content)
        return json.strip()

    async def _rank_name_matches_with_llm(self, restaurant_name: str, legal_records: list[CorporationSearchRecord]) -> list[CorporationSearchRecord]:
        legal_normalized_name_to_record = {}
        for record in legal_records:
            legal_normalized_name_to_record[self._normalize_name(record.corpName)] = record

        try:
            prompt = f"""
            You are a business name matching expert. Given a target business name and candidate matches with fuzzy scores, return the top 3 most likely matches.

Target Name: {restaurant_name}

Candidates:
{legal_normalized_name_to_record.keys()}

Consider:
- Core business name after removing legal suffixes (Inc, LLC, Corp, S.A., etc.)
- Abbreviations: Intl ↔ International, Assoc ↔ Associates, Bros ↔ Brothers
- Spanish/English: Compañía ↔ Company, Servicios ↔ Services
- Franchise/store numbers (#1234, Store 5678)
- Ignore: punctuation, capitalization, "the", common words (de, del, of, and)

Return a JSON object with a "matches" key containing an array of the top 3 candidate names:
{{"matches": ["candidate name 1", "candidate name 2", "candidate name 3"]}}
            """

            response = await self.openai_client.chat_completion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            if response and "content" in response:
                response_content = response["content"]
                if response_content:
                    top_matches = self._clean_json_response(response_content)
                    logger.info(f"OpenAI gave top matches: '{restaurant_name}' → '{top_matches}'")
                    top_matches = json.loads(top_matches)
                    top_record = []
                    for legal_name in top_matches["matches"]:
                        top_record.append(legal_normalized_name_to_record[legal_name])
                    logger.info(f"top records: {top_record}'")
                    return top_record
            
            logger.warning(f"OpenAI returned empty response for '{restaurant_name}' for ranking best matches")
            return []
            
        except Exception as e:
            logger.error(f"Error ranking legal names with OpenAI: {e}. ")
            return []

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
        Searches for the top 3 matching business records for a given restaurant.
        """
        normalized_restaurant_name = self._normalize_name(restaurant.name)
        search_records = await self.incorporation_searcher.search_business(
            normalized_restaurant_name,
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
                name_score = self._calculate_name_similarity(normalized_restaurant_name, record.corpName)
                scored_records.append((name_score, record))
        
        # Sort by fuzzy score and take top 10 and then ask llm to give us to 3
        scored_records.sort(key=lambda x: x[0], reverse=True)
        top_10_records = [record for _, record in scored_records[:10]]
        top_3_records = await self._rank_name_matches_with_llm(normalized_restaurant_name, top_10_records)

        if not top_3_records:
            top_3_records = [record for record in top_10_records[:3]]
        
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
