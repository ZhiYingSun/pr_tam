"""
E2E test demonstrating how to mock the searcher using dependency injection.

This test shows the pattern for testing the orchestrator without making real API calls.
The key is that the orchestrator accepts a searcher as a dependency, so we can pass
in a mock implementation for testing.
"""
import pytest
import asyncio
from pathlib import Path
from src.pipelines.orchestrator import PipelineOrchestrator
from src.data.models import MatchingConfig
from src.searchers.async_searcher import AsyncMockIncorporationSearcher


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

