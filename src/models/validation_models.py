"""
Validation models for LLM-based match validation.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class OpenAIValidationResponse(BaseModel):
    """Typed response from OpenAI validation API."""
    match_score: int = Field(..., ge=0, le=100, description="Match score from 0 to 100")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence level")
    recommendation: Literal["accept", "reject", "manual_review"] = Field(..., description="Recommendation")
    reasoning: str = Field(..., description="Explanation of the decision")


class OpenAIMultiCandidateResponse(BaseModel):
    """Typed response from OpenAI validation API for multi-candidate selection."""
    selected_candidate_index: int = Field(..., ge=-1, description="Index (0-based) of the best matching candidate, or -1 if none match")
    match_score: float = Field(..., ge=0, le=100, description="Match score from 0 to 100 for selected candidate (can be float)")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence level")
    recommendation: Literal["accept", "reject", "manual_review"] = Field(..., description="Recommendation")
    reasoning: str = Field(..., description="Brief explanation of why this candidate was selected")


class ValidationResult(BaseModel):
    """Represents the result of an LLM validation for a single match."""
    restaurant_name: str
    business_legal_name: str
    rapidfuzz_confidence_score: float
    openai_match_score: Optional[float] = None
    openai_confidence: Optional[str] = None
    openai_recommendation: Optional[str] = None
    openai_reasoning: Optional[str] = None
    openai_raw_response: Optional[str] = None
    final_status: str = "pending"  # accept, reject, manual_review
    # Fields for multi-candidate validation
    selected_match_index: Optional[int] = None  # Index of selected match from candidates (0-based)
    total_candidates_evaluated: Optional[int] = None  # Total number of candidates evaluated
    # Restaurant details for final output
    restaurant_google_id: Optional[str] = None
    restaurant_address: Optional[str] = None
    restaurant_city: Optional[str] = None
    restaurant_postal_code: Optional[str] = None
    restaurant_website: Optional[str] = None
    restaurant_phone: Optional[str] = None
    restaurant_rating: Optional[float] = None
    restaurant_reviews_count: Optional[int] = None
    restaurant_main_type: Optional[str] = None
    # Business details for final output
    business_registration_index: Optional[str] = None

