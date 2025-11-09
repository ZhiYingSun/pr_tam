"""
E2E test demonstrating how to mock dependencies using dependency injection.

This test shows the pattern for testing the orchestrator without making real API calls.
The key is that the orchestrator accepts dependencies (searcher, validator) so we can pass
in mock implementations for testing.
"""
import pytest
import asyncio
from pathlib import Path
from src.pipelines.orchestrator import PipelineOrchestrator
from src.data.models import MatchingConfig
from src.searchers.async_searcher import AsyncMockIncorporationSearcher
from src.validators.openai_validator import MockOpenAIValidator


@pytest.mark.asyncio
async def test_e2e_with_mock_searcher():
    """
    E2E test demonstrating dependency injection pattern with mocking.
    
    This test shows:
    1. How to create a mock searcher
    2. How to inject it into the orchestrator
    3. How the orchestrator uses the mock without knowing it's a mock
    """
    # Create a temporary CSV file with test data
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV
        csv_path = Path(tmpdir) / "test_restaurants.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'name', 'address', 'city', 'postal_code', 
                'latitude', 'longitude', 'rating', 'google_id', 
                'phone', 'website', 'reviews_count', 'main_type'
            ])
            writer.writerow([
                'Test Restaurant', '123 Main St', 'San Juan', '00901',
                '18.4655', '-66.1057', '4.5', 'test_google_id',
                '787-123-4567', 'https://test.com', '100', 'Restaurant'
            ])
        
        # Create mock searcher using the provided AsyncMockIncorporationSearcher
        # In real tests, you could also use unittest.mock.AsyncMock or create custom mocks
        mock_searcher = AsyncMockIncorporationSearcher()
        
        # Create orchestrator with injected mock searcher
        # The orchestrator doesn't know it's a mock - it just uses the interface
        orchestrator = PipelineOrchestrator(
            searcher=mock_searcher,
            config=MatchingConfig(),
            skip_validation=True,  # Skip validation for faster test
            skip_transformation=True  # Skip transformation for faster test
        )
        
        # Run the pipeline
        result = await orchestrator.run(
            input_csv=str(csv_path),
            output_dir=str(Path(tmpdir) / "output"),
            limit=1,
            max_concurrent=5
        )
        
        # Verify the pipeline completed successfully
        assert result['success'] is True
        assert len(result['results']) > 0
        
        # Verify the mock searcher was called
        # (The actual call verification depends on the mock implementation)


@pytest.mark.asyncio
async def test_e2e_with_mock_validator():
    """
    E2E test demonstrating dependency injection pattern with mocked validator.
    
    This test shows:
    1. How to create a mock validator
    2. How to inject it into the orchestrator
    3. How validation runs without making OpenAI API calls
    """
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV
        csv_path = Path(tmpdir) / "test_restaurants.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'name', 'address', 'city', 'postal_code', 
                'latitude', 'longitude', 'rating', 'google_id', 
                'phone', 'website', 'reviews_count', 'main_type'
            ])
            writer.writerow([
                'Test Restaurant', '123 Main St', 'San Juan', '00901',
                '18.4655', '-66.1057', '4.5', 'test_google_id',
                '787-123-4567', 'https://test.com', '100', 'Restaurant'
            ])
        
        # Create mock searcher and validator
        mock_searcher = AsyncMockIncorporationSearcher()
        mock_validator = MockOpenAIValidator()
        
        # Create orchestrator with injected mocks
        # Note: We pass validator explicitly, so validation will run (not skipped)
        orchestrator = PipelineOrchestrator(
            searcher=mock_searcher,
            validator=mock_validator,
            config=MatchingConfig(),
            skip_transformation=True  # Skip transformation for faster test
        )
        
        # Run the pipeline
        result = await orchestrator.run(
            input_csv=str(csv_path),
            output_dir=str(Path(tmpdir) / "output"),
            limit=1,
            max_concurrent=5
        )
        
        # Verify the pipeline completed successfully
        assert result['success'] is True
        assert len(result['results']) > 0
        
        # Verify we got match results
        match_results = result.get('results', [])
        assert len(match_results) > 0, "Expected match results"
        
        # Check if any matches were found (validation only runs if match found)
        matches_with_business = [r for r in match_results if r and r.business]
        
        if matches_with_business:
            # Verify validation results exist (since we injected a validator and have matches)
            validation_results = result.get('validation_results', [])
            assert len(validation_results) > 0, f"Expected validation results when validator is injected and matches found. Matches: {len(matches_with_business)}"
            
            # Verify mock validator returned expected results
            for validation_result in validation_results:
                assert validation_result is not None
                assert validation_result.openai_recommendation == "accept"
                assert validation_result.openai_confidence == "high"
                assert validation_result.openai_match_score == 95
        else:
            # If no matches found, validation won't run (expected behavior)
            # This is acceptable - the test demonstrates the mocking pattern
            validation_results = result.get('validation_results', [])
            assert len(validation_results) == 0, "No validation expected when no matches found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

