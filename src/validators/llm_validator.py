"""
LLM Validator - Business logic for validating matches using LLM.
Uses OpenAIClient via dependency injection.
"""
import asyncio
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from openai import APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data.models import MatchResult, MatchingConfig
from src.clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Represents the result of an LLM validation for a single match."""
    restaurant_name: str
    business_legal_name: str
    rapidfuzz_confidence_score: float
    openai_match_score: Optional[int] = None
    openai_confidence: Optional[str] = None
    openai_recommendation: Optional[str] = None
    openai_reasoning: Optional[str] = None
    openai_raw_response: Optional[str] = None
    final_status: str = "pending" # accept, reject, manual_review

class LLMValidator:
    """
    Validates restaurant-business matches using OpenAI's language model.
    Uses OpenAIClient via dependency injection for making API calls.
    """
    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_concurrent_calls: int = 5
    ):
        """
        Initialize LLM validator.
        
        Args:
            openai_client: OpenAIClient instance (dependency injection)
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_concurrent_calls: Maximum concurrent validation calls
        """
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
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
    async def _call_openai_api(self, prompt: str) -> Optional[Dict]:
        """Makes an asynchronous call to the OpenAI API via OpenAIClient."""
        try:
            response = await self.openai_client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            return response
        except APIStatusError as e:
            logger.error(f"OpenAI API error (status {e.status_code}): {e.response}")
            raise  # Re-raise to trigger retry
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
            return None

    async def validate_match(self, match: MatchResult) -> ValidationResult:
        """Validates a single match using OpenAI."""
        async with self.semaphore:
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
                openai_raw_response=json.dumps(openai_response) if openai_response else None
            )

            if openai_response:
                validation_result.openai_match_score = openai_response.get("match_score")
                validation_result.openai_confidence = openai_response.get("confidence")
                validation_result.openai_recommendation = openai_response.get("recommendation")
                validation_result.openai_reasoning = openai_response.get("reasoning")
                validation_result.final_status = openai_response.get("recommendation", "manual_review")
            else:
                validation_result.openai_recommendation = "manual_review"
                validation_result.openai_reasoning = "OpenAI response invalid or failed."
                validation_result.final_status = "manual_review"

            return validation_result

    async def validate_matches_batch(self, matches: List[MatchResult], batch_size: int = 10) -> List[ValidationResult]:
        """Validates a list of matches in batches."""
        all_validation_results: List[ValidationResult] = []
        
        for i in range(0, len(matches), batch_size):
            batch_matches = matches[i:i + batch_size]
            logger.info(f"Processing validation batch {i // batch_size + 1}/{len(matches) // batch_size + 1}")
            
            tasks = [self.validate_match(match) for match in batch_matches]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
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


def create_validation_summary(validation_results: List[ValidationResult]) -> Dict[str, Any]:
    """Generates a summary of the validation results."""
    total_validated = len(validation_results)
    accepted = [r for r in validation_results if r.openai_recommendation == "accept"]
    rejected = [r for r in validation_results if r.openai_recommendation == "reject"]
    manual_review = [r for r in validation_results if r.openai_recommendation == "manual_review"]

    avg_openai_score = sum([r.openai_match_score for r in validation_results if r.openai_match_score is not None]) / len([r for r in validation_results if r.openai_match_score is not None]) if [r for r in validation_results if r.openai_match_score is not None] else 0
    avg_rapidfuzz_score = sum([r.rapidfuzz_confidence_score for r in validation_results]) / total_validated if total_validated else 0

    return {
        "total_validated": total_validated,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "manual_review_count": len(manual_review),
        "accepted_percentage": (len(accepted) / total_validated * 100) if total_validated else 0,
        "rejected_percentage": (len(rejected) / total_validated * 100) if total_validated else 0,
        "manual_review_percentage": (len(manual_review) / total_validated * 100) if total_validated else 0,
        "avg_openai_score": avg_openai_score,
        "avg_rapidfuzz_score": avg_rapidfuzz_score,
        "high_confidence_openai": len([r for r in validation_results if r.openai_confidence == "high"]),
        "medium_confidence_openai": len([r for r in validation_results if r.openai_confidence == "medium"]),
        "low_confidence_openai": len([r for r in validation_results if r.openai_confidence == "low"]),
        "sample_accepts": accepted[:2],
        "sample_rejects": rejected[:2],
        "sample_manual_reviews": manual_review[:2]
    }

