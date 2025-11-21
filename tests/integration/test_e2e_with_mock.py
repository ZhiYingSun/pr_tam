"""
E2E test demonstrating dependency injection with mock implementations.

This test shows proper dependency injection where mock clients (Zyte and OpenAI)
are injected into the components, allowing the real business logic to run
while avoiding actual API calls.
"""
import pytest
import asyncio

from src.models.models import (
    RestaurantRecord,
    MatchingConfig,
)
from src.orchestrator.orchestrator import PipelineOrchestrator
from src.searchers.searcher import IncorporationSearcher
from src.utils.loader import CSVRestaurantLoader
from src.utils.report_generator import ReportGenerator
from tests.fixtures.mock_clients import MockZyteClient, MockOpenAIClient


@pytest.mark.asyncio
async def test_e2e_with_mock_clients():
    """
    E2E test using dependency injection with mock Zyte and OpenAI clients.
    
    This demonstrates proper dependency injection:
    1. Mock clients implement their respective protocols
    2. Mocks are injected via constructors (no monkey patching)
    3. Real business logic runs with mocked external API calls
    4. Clean, testable architecture
    """
    # Create a test restaurant
    restaurant = RestaurantRecord(
        name="Test Restaurant",
        address="123 Main St",
        city="San Juan",
        postal_code="00901",
        coordinates=(-66.1, 18.5),
        rating=4.5
    )
    
    # Create mock clients with predefined responses
    mock_zyte_client = MockZyteClient(api_key="test_key")
    mock_openai_client = MockOpenAIClient(api_key="test_key")
    
    # Inject mock Zyte client into searcher
    searcher = IncorporationSearcher(
        zyte_api_key="test_key",
        zyte_client=mock_zyte_client
    )
    
    # Create orchestrator with injected dependencies
    loader = CSVRestaurantLoader()
    report_generator = ReportGenerator()
    orchestrator = PipelineOrchestrator(
        openai_client=mock_openai_client,
        searcher=searcher,
        config=MatchingConfig(),
        loader=loader,
        report_generator=report_generator
    )
    
    # Process the restaurant
    async with searcher:
        from src.matchers.matcher import RestaurantMatcher
        matcher = RestaurantMatcher(searcher, mock_openai_client)
        
        match_result, validation_result = await orchestrator.process_restaurant(restaurant, matcher)
        
        # Verify that the mocks were called
        assert mock_zyte_client.post_call_count > 0, "Zyte search request should have been made"
        assert len(mock_zyte_client.post_requests) > 0, "Should have recorded Zyte POST requests"
        
        # The match result may or may not be found depending on matching logic
        # But the test passes if the flow completes without errors
        print(f"\n=== E2E Test Results ===")
        print(f"Mock Zyte POST calls: {mock_zyte_client.post_call_count}")
        print(f"Mock Zyte GET calls: {mock_zyte_client.get_call_count}")
        print(f"Mock OpenAI calls: {mock_openai_client.call_count}")
        print(f"Match result: {match_result}")
        print(f"Validation result: {validation_result}")


@pytest.mark.asyncio
async def test_mock_clients_custom_responses():
    """
    Test that we can customize mock responses for different test scenarios.
    """
    # Create mock clients
    mock_zyte_client = MockZyteClient()
    mock_openai_client = MockOpenAIClient()
    
    # Customize the Zyte search response
    custom_zyte_response = {
        "response": {
            "records": [
                {
                    "businessEntityId": 99999,
                    "registrationNumber": 99999,
                    "registrationIndex": "99999-999",
                    "corpName": "Custom Mock Corp",
                    "statusEn": "ACTIVE",
                    "statusEs": "ACTIVA"
                }
            ]
        },
        "code": 1,
        "info": None,
        "success": True
    }
    mock_zyte_client.set_search_response(custom_zyte_response)
    
    # Customize the OpenAI validation response
    custom_openai_response = {
        "is_match": False,
        "confidence": "low",
        "reasoning": "Custom mock - names do not match"
    }
    mock_openai_client.set_response(custom_openai_response)
    
    # Create searcher with custom mock
    searcher = IncorporationSearcher(
        zyte_api_key="test_key",
        zyte_client=mock_zyte_client
    )
    
    # Test the search
    async with searcher:
        records = await searcher.search_business("Custom Business")
        
        # Verify custom response was used
        assert len(records) == 1
        assert records[0].corpName == "Custom Mock Corp"
        assert records[0].businessEntityId == 99999
        assert mock_zyte_client.post_call_count == 1
        
        # Test OpenAI mock
        response = await mock_openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.2
        )
        
        assert response["is_match"] is False
        assert response["confidence"] == "low"
        assert mock_openai_client.call_count == 1
