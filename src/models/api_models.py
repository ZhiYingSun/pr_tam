"""
API response models for external API integrations (Zyte, Puerto Rico incorporation API).
"""
import json
import base64
import binascii
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


class ZyteHttpResponse(BaseModel):
    """Response model for Zyte API HTTP responses."""
    httpResponseBody: Optional[str] = Field(None, description="Base64 encoded response body")

    def decode_body(self) -> Dict[str, Any]:
        """Decode base64 response body and parse JSON."""
        if not self.httpResponseBody:
            raise ValueError("httpResponseBody is missing or empty")

        try:
            decoded_body = base64.b64decode(self.httpResponseBody).decode('utf-8')
            return json.loads(decoded_body)
        except (binascii.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode base64 response body: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from decoded body: {e}")


# Corporation Search API Response Models
class CorporationSearchRecord(BaseModel):
    """Single record from corporation search API."""
    businessEntityId: Optional[int] = None
    registrationNumber: Optional[int] = None
    registrationIndex: Optional[str] = None
    corpName: Optional[str] = None
    classEs: Optional[str] = None
    classEn: Optional[str] = None
    profitTypeEs: Optional[str] = None
    profitTypeEn: Optional[str] = None
    statusId: Optional[int] = None
    statusEs: Optional[str] = None
    statusEn: Optional[str] = None


class CorporationSearchResponseData(BaseModel):
    """Data wrapper for corporation search response."""
    totalRecords: Optional[int] = None
    records: List[CorporationSearchRecord] = Field(default_factory=list)


class CorporationSearchResponse(BaseModel):
    """Complete corporation search API response."""
    response: Optional[CorporationSearchResponseData] = None
    code: Optional[int] = None
    info: Optional[Any] = None
    success: Optional[bool] = None


# Corporation Detail API Response Models
class StreetAddressDetail(BaseModel):
    """Street address detail from corporation detail API."""
    address1: Optional[str] = None
    address2: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None


class IndividualNameDetail(BaseModel):
    """Individual name detail from corporation detail API."""
    firstName: Optional[str] = None
    middleName: Optional[str] = None
    lastName: Optional[str] = None
    surName: Optional[str] = None


class OrganizationNameDetail(BaseModel):
    """Organization name detail from corporation detail API."""
    name: Optional[str] = None


class CorporationDetail(BaseModel):
    """Corporation detail from corporation detail API."""
    corpName: Optional[str] = None
    corpRegisterNumber: Optional[int] = None
    corpRegisterIndex: Optional[str] = None
    statusEn: Optional[str] = None


class MainLocationDetail(BaseModel):
    """Main location detail from corporation detail API."""
    streetAddress: Optional[StreetAddressDetail] = None


class ResidentAgentDetail(BaseModel):
    """Resident agent detail from corporation detail API."""
    isIndividual: Optional[bool] = None
    individualName: Optional[IndividualNameDetail] = None
    organizationName: Optional[OrganizationNameDetail] = None
    streetAddress: Optional[StreetAddressDetail] = None


class CorporationDetailResponseData(BaseModel):
    """Data wrapper for corporation detail response."""
    corporation: Optional[CorporationDetail] = None
    mainLocation: Optional[MainLocationDetail] = None
    residentAgent: Optional[ResidentAgentDetail] = None


class CorporationDetailResponse(BaseModel):
    """Complete corporation detail API response."""
    response: Optional[CorporationDetailResponseData] = None
    code: Optional[int] = None
    info: Optional[Any] = None
    success: Optional[bool] = None

