"""
LLM Validator - Business logic for validating matches using LLM.
"""
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from pydantic import ValidationError
from openai import APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.models import MatchResult, MatchingConfig
from src.models.validation_models import OpenAIValidationResponse, ValidationResult
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

    async def validate_matches_batch(self, matches: List[MatchResult], batch_size: int = 10) -> List[ValidationResult]:
        """Validates a list of matches in batches."""
        all_validation_results: List[ValidationResult] = []
        total_batches = (len(matches) + batch_size - 1) // batch_size
        
        for i in range(0, len(matches), batch_size):
            batch_matches = matches[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Processing validation batch {batch_num}/{total_batches}")
            
            tasks = [self.validate_match(match) for match in batch_matches]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count errors in this batch
            batch_error_count = sum(1 for r in batch_results if isinstance(r, Exception))
            batch_error_rate = batch_error_count / len(batch_matches) if batch_matches else 0.0
            
            # Check if batch failure rate exceeds threshold
            if batch_error_rate > 0.5:
                raise RuntimeError(
                    f"Validation batch {batch_num} failed: {batch_error_count}/{len(batch_matches)} errors "
                    f"({batch_error_rate*100:.1f}% error rate exceeds 50% threshold)"
                )
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error validating match for '{batch_matches[j].restaurant.name}': {result}")
                    # Create a default ValidationResult for errors
                    error_validation = ValidationResult(
                        restaurant_name=batch_matches[j].restaurant.name,
                        business_legal_name=batch_matches[j].business.legal_name if batch_matches[j].business else "",
                        rapidfuzz_confidence_score=batch_matches[j].confidence_score,
                        openai_recommendation="manual_review",
                        openai_reasoning=f"Error during OpenAI validation: {result}",
                        final_status="manual_review"
                    )
                    all_validation_results.append(error_validation)
                else:
                    all_validation_results.append(result)
        
        logger.info(f"Completed batch validation: {len(all_validation_results)} results")
        return all_validation_results

