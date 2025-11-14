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
        prompt = f"""You are an expert business entity matching system for Puerto Rico. Determine if a business (Google Maps) and a business entity (PR incorporation records) represent the same real-world entity.

## INPUT DATA

### Business (Google Maps)
- Name: {restaurant.name}
- Address: {restaurant.address}
- City: {restaurant.city}
- Postal Code: {restaurant.postal_code}
- Category: {restaurant.main_type}

### Business Entity (PR Registry)
- Legal Name: {business.legal_name}
- Registration #: {business.registration_number}
- Address: {business.business_address}
- Status: {business.status}
- Resident Agent: {business.resident_agent_name}
- Agent Address: {business.resident_agent_address}

### Preliminary Analysis
- RapidFuzz Score: {match.confidence_score:.1f}%
- Match Reason: {match.match_reason}

## EVALUATION CRITERIA

### 1. Name Matching (Weight: 40%)

**Puerto Rico Legal Suffixes to Strip/Normalize:**
- English: LLC, Inc, Corp, Co, Corporation, Limited, LTD
- Spanish: SRL, SA, Corp., Corporación, Ltda, Limitada, Sociedad, S.E.
- Common: SE (Sociedad Especial), Corp. SE, Inc. SE

**Name Matching Rules:**
- Ignore accents/diacritics: "María" = "Maria", "José" = "Jose", "Café" = "Cafe"
- Common Spanish abbreviations: "Cía" = "Compañía", "Asoc." = "Asociación", "Serv." = "Servicios"
- Articles: "El", "La", "Los", "Las" (may be included/excluded)
- Possessives: "De" (e.g., "Tienda De Maria" vs "Tienda Maria")
- Word order variations: "Centro Puerto Rico" vs "Puerto Rico Centro"
- Ampersand vs "Y": "&" = "y" = "Y"
- Numbers: "2" = "Two" = "Dos"
- Common business descriptors may be added/removed: "Shop", "Store", "Center", "Tienda", "Centro"

**Red Flags:**
- Completely different base names
- Owner names don't match business names
- English vs Spanish names with no clear connection

### 2. Address Verification (Weight: 35%)

**Puerto Rico Address Format Considerations:**

**Street Types (normalize these):**
- Calle / C. / C = Street
- Avenida / Ave. / Av. = Avenue  
- Carretera / Carr. / CR = Highway/Road
- Urbanización / Urb. = Housing development
- Residencial / Res. = Residential area
- Parcela / Parc. = Parcel
- Camino / Cam. = Road/Path
- Boulevard / Blvd. / Bulevar

**Address Components:**
- Street number may come before or after street name
- Examples: "123 Calle Luna", "Calle Luna 123", "C. Luna #123"
- Building/Suite: "Edificio", "Edif.", "Suite", "Local", "Apto", "Oficina"
- Interior identifiers: "Int.", "Interior", "#"

**Geographic Context:**
- Puerto Rico uses US postal codes (00600-00799, 00900-00999)
- Major metro areas: San Juan (009xx), Ponce (007xx-008xx), Mayagüez (006xx-008xx)
- Same postal code = strong signal
- Adjacent postal codes = possible match
- **Important**: Resident agent addresses are often NOT the business location (commonly law offices in San Juan)

**Matching Logic:**
1. **Exact match** (ignoring abbreviations/format): Highest confidence
2. **Same street + number, different unit/local**: Very high confidence
3. **Same Urbanización/area + same street**: High confidence
4. **Same postal code + similar street name**: Moderate confidence
5. **Same municipality, different postal code**: Low confidence
6. **Resident agent address only**: Very low confidence (likely mismatch)

**Red Flags:**
- Different municipalities (e.g., San Juan vs Ponce)
- Non-adjacent postal codes with no other address info
- Only resident agent address matches (unless it's explicitly the business address)

### 3. Business Type Alignment (Weight: 15%)

Evaluate if the Google Maps category is compatible with the type of business entity that would be registered.

**General Compatibility:**
- Retail stores → Commercial, Retail, Sales, Ventas
- Professional services → Services, Servicios profesionales, Consulting
- Food/Hospitality → Food service, Hospitality, Hostelería
- Healthcare → Medical, Health services, Servicios de salud
- Construction → Construction, Construcción, Contractor
- Real estate → Real estate, Bienes raíces, Property management

**Consider:**
- Generic entity types (Commercial, Services, Comercial, Servicios) are compatible with most businesses
- DBAs (doing business as) may operate under different names than legal entity
- Holding companies may own various business types

**Red Flags:**
- Clear incompatibility (e.g., Google Maps shows "Dentist" but legal entity is "Construction Company")
- Business category suggests individual professional (doctor, lawyer) but entity is unrelated corporation

### 4. Operational Status (Weight: 10%)

- **Active/Activa**: Normal operation
- **Inactive/Inactiva**: Possible mismatch
- **Dissolved/Disuelta**: Likely mismatch (unless <6 months)
- **Suspended/Suspendida**: Verify carefully

## OUTPUT REQUIREMENTS

Return ONLY valid JSON (no markdown fences, no preamble):

{{
    "match_score": <integer 0-100>,
    "confidence": "<high|medium|low>",
    "recommendation": "<accept|manual_review|reject>",
    "reasoning": "<concise explanation citing specific factors>"
}}

**Recommendation Thresholds:**
- `accept`: match_score ≥ 85 AND confidence = "high"
- `manual_review`: match_score 40-84 OR confidence = "medium"  
- `reject`: match_score < 40 AND confidence = "low"

**Confidence Levels:**
- `high`: Multiple strong confirming signals, no major conflicts
- `medium`: Some confirming signals, minor conflicts or missing data
- `low`: Weak signals, major conflicts, or insufficient data
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

