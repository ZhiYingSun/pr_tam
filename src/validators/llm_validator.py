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
    OpenAIMultiCandidateResponse,
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


    def _construct_multi_candidate_prompt(self, restaurant: RestaurantRecord, matches: List[MatchResult]) -> str:
        """
        Constructs a simple prompt for OpenAI to evaluate 25 candidate matches and select the best one.
        
        Args:
            restaurant: RestaurantRecord to match
            matches: List of MatchResult candidates (up to 25, sorted by confidence score, already filtered to have business records)
            
        Returns:
            Formatted prompt string
        """
        # Build simplified candidate list - just key info
        candidates_text = []
        for idx, match in enumerate(matches):
            business = match.business
            candidates_text.append(
                f"#{idx + 1}. {business.legal_name} | "
                f"Address: {business.business_address or 'N/A'} | "
                f"Score: {match.confidence_score:.1f}% | "
                f"Postal: {'Yes' if match.postal_code_match else 'No'} | "
                f"City: {'Yes' if match.city_match else 'No'}"
            )

        # TODO: tune this
        prompt = f"""You are evaluating 25 business candidates to find the best match for a restaurant.

Restaurant: {restaurant.name}
Location: {restaurant.address}, {restaurant.city} {restaurant.postal_code}
Type: {restaurant.main_type}

Candidates (sorted by initial score):
{chr(10).join(candidates_text)}

Task: Select the ONE candidate that best matches the restaurant, or indicate none match.

Consider: name similarity (accounting for LLC/Inc suffixes), address proximity, and business type compatibility.

Respond in JSON:
{{
    "selected_candidate_index": <0-24 or -1 if none>,
    "match_score": <0-100>,
    "confidence": "high|medium|low",
    "recommendation": "accept|reject|manual_review",
    "reasoning": "<brief explanation of why this candidate was selected>"
}}"""
        return prompt

    @retry(
        stop=stop_after_attempt(MatchingConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(APIStatusError)
    )
    async def _make_multi_candidate_llm_call(self, prompt: str) -> Optional[OpenAIMultiCandidateResponse]:
        """
        Makes an asynchronous call to the OpenAI API for multi-candidate validation.
        
        Returns:
            OpenAIMultiCandidateResponse if successful, None on error
        """
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
                return OpenAIMultiCandidateResponse(**response_dict)
            except ValidationError as e:
                logger.error(f"Failed to validate OpenAI multi-candidate response structure: {e}. Response: {response_dict}")
                return None
            
        except APIStatusError as e:
            logger.error(f"OpenAI API error (status {e.status_code}): {e.response}")
            raise  # Re-raise to trigger retry
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
            return None

    async def validate_best_match_from_candidates(
        self, 
        matches: List[MatchResult]
    ) -> Tuple[Optional[MatchResult], ValidationResult]:
        """
        Validates multiple candidate matches (up to 25) and selects the best one using OpenAI.
        
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

        prompt = self._construct_multi_candidate_prompt(restaurant, valid_matches)
        llm_response = await self._make_multi_candidate_llm_call(prompt)
        
        if not llm_response:
            logger.error(f"OpenAI validation failed for '{restaurant.name}'")
            # Return the top match by confidence score as fallback
            top_match = valid_matches[0]
            return top_match, ValidationResult(
                restaurant_name=restaurant.name,
                business_legal_name=top_match.business.legal_name if top_match.business else "",
                rapidfuzz_confidence_score=top_match.confidence_score,
                openai_recommendation="manual_review",
                openai_reasoning="OpenAI validation failed. Using top confidence score match as fallback.",
                final_status="manual_review",
                selected_match_index=0,
                total_candidates_evaluated=len(valid_matches),
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
                business_registration_index=top_match.business.registration_index if top_match.business else None
            )
        
        # Validate selected index
        selected_index = llm_response.selected_candidate_index
        
        if selected_index == -1:
            # LLM determined none of the candidates match
            logger.info(f"LLM determined no good match for '{restaurant.name}' among {len(valid_matches)} candidates")
            return None, ValidationResult(
                restaurant_name=restaurant.name,
                business_legal_name="",
                rapidfuzz_confidence_score=0.0,
                openai_match_score=llm_response.match_score if llm_response.match_score is not None else None,
                openai_confidence=llm_response.confidence,
                openai_recommendation=llm_response.recommendation,
                openai_reasoning=llm_response.reasoning,
                final_status=llm_response.recommendation,
                selected_match_index=-1,
                total_candidates_evaluated=len(valid_matches),
                openai_raw_response=llm_response.model_dump_json(),
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
        
        if selected_index < 0 or selected_index >= len(valid_matches):
            logger.warning(
                f"Invalid selected_index {selected_index} for '{restaurant.name}'. "
                f"Using top match (index 0) as fallback."
            )
            selected_index = 0
        
        # Get the selected match
        selected_match = valid_matches[selected_index]

        # Create validation result with all restaurant and business details
        validation_result = ValidationResult(
            restaurant_name=restaurant.name,
            business_legal_name=selected_match.business.legal_name if selected_match.business else "",
            rapidfuzz_confidence_score=selected_match.confidence_score,
            openai_match_score=llm_response.match_score if llm_response.match_score is not None else None,
            openai_confidence=llm_response.confidence,
            openai_recommendation=llm_response.recommendation,
            openai_reasoning=llm_response.reasoning,
            final_status=llm_response.recommendation,
            selected_match_index=selected_index,
            total_candidates_evaluated=len(valid_matches),
            openai_raw_response=llm_response.model_dump_json(),
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
            business_registration_index=selected_match.business.registration_index if selected_match.business else None
        )
        
        logger.info(
            f"LLM selected candidate #{selected_index + 1} for '{restaurant.name}': "
            f"'{selected_match.business.legal_name if selected_match.business else 'N/A'}' "
            f"(recommendation: {llm_response.recommendation}, "
            f"score: {llm_response.match_score}%)"
        )
        
        return selected_match, validation_result

