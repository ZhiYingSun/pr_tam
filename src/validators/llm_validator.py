"""
LLM Validator - Business logic for validating matches using LLM.
"""
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from pydantic import ValidationError
from openai import APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.models import MatchResult, MatchingConfig, RestaurantRecord
from src.models.validation_models import (
    OpenAIValidationResponse,
    ValidationResult
)
from src.clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class LLMValidator:
    """
    Validates restaurant-business matches using a language model.
    """
    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2
    ):
        """
        Initialize LLM validator.
        
        Args:
            openai_client: OpenAIClient instance
            model: Model name
            temperature: Sampling temperature
        """
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        logger.info(f"Initialized LLM validator with model: {self.model}")

    def _construct_prompt(self, match: MatchResult) -> str:
        """Constructs the prompt for OpenAI based on the match details."""
        restaurant = match.restaurant
        business = match.business
        # TODO: tune this
        prompt = f"""
Context: You are an expert in business data matching for Puerto Rico. Your task is to determine if a given restaurant (from Google Maps) and a business entity (from Puerto Rico incorporation documents) represent the same real-world entity.

Restaurant Information (Google Maps):
- Name: {restaurant.name}
- Address: {restaurant.address}
- City: {restaurant.city}
- Postal Code: {restaurant.postal_code}
- Main Type: {restaurant.main_type}

Business Information (Puerto Rico Incorporation Documents):
- Legal Name: {business.legal_name}
- Registration Number: {business.registration_number}
- Business Address: {business.business_address}
- Status: {business.status}
- Resident Agent Name: {business.resident_agent_name}
- Resident Agent Address: {business.resident_agent_address}

RapidFuzz Match Details:
- Confidence Score: {match.confidence_score:.1f}%
- Match Reason: {match.match_reason}

Task: Evaluate the likelihood that the "Restaurant Information" and "Business Information" refer to the same entity.

Consider the following factors:
1.  **Name Similarity**: How closely do the names match, accounting for legal suffixes (e.g., LLC, Inc.), common abbreviations, and slight variations?
2.  **Address Proximity**: Are the addresses identical or very close? Consider if one might be a mailing address and the other a physical location, or if they are in the same city/postal code.
3.  **Business Type/Purpose**: Is the restaurant's main type compatible with the business's legal purpose or general nature?
4.  **Status**: Is the business active?

Provide your response in a JSON format with the following keys:
-   `match_score`: An integer from 0 to 100 representing the likelihood of a match.
-   `confidence`: A string, one of "high", "medium", or "low".
-   `recommendation`: A string, one of "accept", "reject", or "manual_review".
-   `reasoning`: A concise string explaining your decision, referencing the factors above.

Example JSON Output:
```json
{{
    "match_score": 90,
    "confidence": "high",
    "recommendation": "accept",
    "reasoning": "Names are very similar, addresses are identical, and business types are compatible."
}}
```
"""
        return prompt

    @retry(
        stop=stop_after_attempt(MatchingConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(APIStatusError)
    )
    async def _call_openai_api(self, prompt: str) -> Optional[OpenAIValidationResponse]:
        """
        Makes an asynchronous call to the OpenAI API via OpenAIClient.
        
        Returns:
            OpenAIValidationResponse if successful, None on error
        """
        # TODO Cache: cache OpenAI validation responses
        # Cache key: hash(restaurant_name + business_legal_name + confidence_score)
        
        try:
            response_dict = await self.openai_client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            if not response_dict:
                return None
            
            try:
                return OpenAIValidationResponse(**response_dict)
            except ValidationError as e:
                logger.error(f"Failed to validate OpenAI response structure: {e}. Response: {response_dict}")
                return None
            
        except APIStatusError as e:
            logger.error(f"OpenAI API error (status {e.status_code}): {e.response}")
            raise  # Re-raise to trigger retry
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
            return None

    async def validate_match(self, match: MatchResult) -> ValidationResult:
        """Validates a single match using OpenAI. Rate limiting handled by OpenAIClient."""
        if not match.business:
            logger.warning(f"Skipping OpenAI validation for '{match.restaurant.name}' due to missing business record.")
            return ValidationResult(
                restaurant_name=match.restaurant.name,
                business_legal_name="",
                rapidfuzz_confidence_score=match.confidence_score,
                openai_recommendation="reject",
                openai_reasoning="No business record available for validation.",
                final_status="reject"
            )

        prompt = self._construct_prompt(match)
        openai_response = await self._call_openai_api(prompt)

        validation_result = ValidationResult(
            restaurant_name=match.restaurant.name,
            business_legal_name=match.business.legal_name,
            rapidfuzz_confidence_score=match.confidence_score,
            openai_raw_response=openai_response.model_dump_json() if openai_response else None
        )

        if openai_response:
            validation_result.openai_match_score = openai_response.match_score
            validation_result.openai_confidence = openai_response.confidence
            validation_result.openai_recommendation = openai_response.recommendation
            validation_result.openai_reasoning = openai_response.reasoning
            validation_result.final_status = openai_response.recommendation
        else:
            validation_result.openai_recommendation = "manual_review"
            validation_result.openai_reasoning = "OpenAI response invalid or failed."
            validation_result.final_status = "manual_review"

        return validation_result


    async def validate_best_match_from_candidates(
        self, 
        matches: List[MatchResult]
    ) -> Tuple[Optional[MatchResult], ValidationResult]:
        """
        Validates multiple candidate matches (up to 25) individually and selects the best one.
        
        Args:
            matches: List of MatchResult candidates (up to 25, sorted by confidence score)
            
        Returns:
            Tuple of (selected MatchResult, ValidationResult). 
            MatchResult may be None if no valid match is found or all candidates are rejected.
        """
        if not matches:
            logger.warning("No matches provided for validation")
            return None, ValidationResult(
                restaurant_name="",
                business_legal_name="",
                rapidfuzz_confidence_score=0.0,
                openai_recommendation="reject",
                openai_reasoning="No candidates provided for validation.",
                final_status="reject",
                total_candidates_evaluated=0
            )
        
        restaurant = matches[0].restaurant
        
        # Filter out matches without business records
        valid_matches = [m for m in matches if m.business is not None]
        
        if not valid_matches:
            logger.warning(f"No valid business records found for '{restaurant.name}'")
            return None, ValidationResult(
                restaurant_name=restaurant.name,
                business_legal_name="",
                rapidfuzz_confidence_score=0.0,
                openai_recommendation="reject",
                openai_reasoning="No valid business records available for validation.",
                final_status="reject",
                total_candidates_evaluated=0,
                # Restaurant details
                restaurant_address=restaurant.address,
                restaurant_city=restaurant.city,
                restaurant_postal_code=restaurant.postal_code,
                restaurant_website=restaurant.website,
                restaurant_phone=restaurant.phone,
                restaurant_rating=restaurant.rating,
                restaurant_reviews_count=restaurant.reviews_count,
                restaurant_main_type=restaurant.main_type
            )

        # Validate each candidate individually
        logger.info(f"Validating {len(valid_matches)} candidates for '{restaurant.name}'")
        validation_tasks = [self.validate_match(match) for match in valid_matches]
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        candidate_validations: List[Tuple[MatchResult, ValidationResult]] = []
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"Error validating candidate {i+1} for '{restaurant.name}': {result}")
                # Create a fallback validation result for errors
                match = valid_matches[i]
                error_validation = ValidationResult(
                    restaurant_name=restaurant.name,
                    business_legal_name=match.business.legal_name if match.business else "",
                    rapidfuzz_confidence_score=match.confidence_score,
                    openai_recommendation="manual_review",
                    openai_reasoning=f"Error during validation: {result}",
                    final_status="manual_review",
                    # Restaurant details
                    restaurant_address=restaurant.address,
                    restaurant_city=restaurant.city,
                    restaurant_postal_code=restaurant.postal_code,
                    restaurant_website=restaurant.website,
                    restaurant_phone=restaurant.phone,
                    restaurant_rating=restaurant.rating,
                    restaurant_reviews_count=restaurant.reviews_count,
                    restaurant_main_type=restaurant.main_type,
                    # Business details
                    business_registration_index=match.business.registration_index if match.business else None
                )
                candidate_validations.append((match, error_validation))
            else:
                # Add restaurant and business details to validation result
                match = valid_matches[i]
                result.restaurant_address = restaurant.address
                result.restaurant_city = restaurant.city
                result.restaurant_postal_code = restaurant.postal_code
                result.restaurant_website = restaurant.website
                result.restaurant_phone = restaurant.phone
                result.restaurant_rating = restaurant.rating
                result.restaurant_reviews_count = restaurant.reviews_count
                result.restaurant_main_type = restaurant.main_type
                result.business_registration_index = match.business.registration_index if match.business else None
                result.total_candidates_evaluated = len(valid_matches)
                result.selected_match_index = i
                candidate_validations.append((match, result))
        
        # Select the best match based on validation results
        # Priority: 1) accept recommendations, 2) highest match_score, 3) highest confidence_score
        def score_validation(match_result_pair: Tuple[MatchResult, ValidationResult]) -> Tuple[int, float, float]:
            match, validation = match_result_pair
            # Priority order: accept > manual_review > reject
            priority_map = {"accept": 3, "manual_review": 2, "reject": 1}
            priority = priority_map.get(validation.openai_recommendation or "reject", 1)
            # Use match_score if available, otherwise fall back to rapidfuzz score
            score = validation.openai_match_score if validation.openai_match_score is not None else validation.rapidfuzz_confidence_score
            return (priority, score, validation.rapidfuzz_confidence_score)
        
        # Sort by priority, then by score (descending)
        candidate_validations.sort(key=score_validation, reverse=True)
        
        if not candidate_validations:
            logger.warning(f"No validations completed for '{restaurant.name}'")
            return None, ValidationResult(
                restaurant_name=restaurant.name,
                business_legal_name="",
                rapidfuzz_confidence_score=0.0,
                openai_recommendation="reject",
                openai_reasoning="No validations completed.",
                final_status="reject",
                total_candidates_evaluated=len(valid_matches),
                # Restaurant details
                restaurant_address=restaurant.address,
                restaurant_city=restaurant.city,
                restaurant_postal_code=restaurant.postal_code,
                restaurant_website=restaurant.website,
                restaurant_phone=restaurant.phone,
                restaurant_rating=restaurant.rating,
                restaurant_reviews_count=restaurant.reviews_count,
                restaurant_main_type=restaurant.main_type
            )
        
        # Get the best match
        best_match, best_validation = candidate_validations[0]
        
        logger.info(
            f"Selected best candidate for '{restaurant.name}': "
            f"'{best_match.business.legal_name if best_match.business else 'N/A'}' "
            f"(recommendation: {best_validation.openai_recommendation}, "
            f"score: {best_validation.openai_match_score or best_validation.rapidfuzz_confidence_score:.1f}%)"
        )
        
        return best_match, best_validation

