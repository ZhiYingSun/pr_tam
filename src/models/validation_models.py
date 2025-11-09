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
    final_status: str = "pending"  # accept, reject, manual_review

