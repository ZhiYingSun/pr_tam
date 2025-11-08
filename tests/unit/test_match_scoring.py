import pytest
import asyncio
from src.data.models import RestaurantRecord, BusinessRecord, MatchingConfig
from src.matchers.async_matcher import AsyncRestaurantMatcher
from src.searchers.async_searcher import AsyncMockIncorporationSearcher


class TestMatchScoring:
    """Test match score calculation"""
    
    @pytest.mark.asyncio
    async def test_name_weighted_to_50_percent(self):
        """Test that name similarity contributes exactly 50% of total score"""
        async with AsyncMockIncorporationSearcher() as searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=10)
            
            restaurant = RestaurantRecord(
                name="Test Restaurant",
                address="123 Main St",
                city="San Juan",
                postal_code="00901",
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            business = BusinessRecord(
                legal_name="Test Restaurant",
                registration_number="12345",
                registration_index="12345-111",
                business_address="456 Different St, Ponce, 00717",  # Different city and postal code
                status="ACTIVE",
                resident_agent_name="John Doe",
                resident_agent_address="789 Agent St"
            )
            
            # Calculate score with 100% name match but no location matches
            name_score = matcher._calculate_name_similarity(restaurant.name, business.legal_name)
            score, _, _, _ = matcher._calculate_match_score(restaurant, business, name_score)
            
            # With 100% name match, weighted to 50, no bonuses = 50 total
            assert 45 <= score <= 55, f"Expected ~50% score for perfect name match, got {score}"
            assert score <= 50.0, f"Name-only match should not exceed 50% (got {score})"
    
    @pytest.mark.asyncio
    async def test_full_match_with_all_bonuses(self):
        """Test that perfect match with all bonuses reaches 100"""
        async with AsyncMockIncorporationSearcher() as searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=10)
            
            restaurant = RestaurantRecord(
                name="Perfect Match Restaurant",
                address="123 Main St",
                city="San Juan",
                postal_code="00901",
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            business = BusinessRecord(
                legal_name="Perfect Match Restaurant",
                registration_number="12345",
                registration_index="12345-111",
                business_address="123 Main St, San Juan, 00901",
                status="ACTIVE",
                resident_agent_name="John Doe",
                resident_agent_address="789 Agent St"
            )
            
            name_score = matcher._calculate_name_similarity(restaurant.name, business.legal_name)
            score, _, _, _ = matcher._calculate_match_score(restaurant, business, name_score)
            
            # Perfect name (50) + postal (30) + city (20) = 100
            assert score == 100.0, f"Expected 100% for perfect match, got {score}"
    
    @pytest.mark.asyncio
    async def test_postal_code_bonus(self):
        """Test postal code bonus adds 30 points"""
        async with AsyncMockIncorporationSearcher() as searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=10)
            
            restaurant = RestaurantRecord(
                name="Different Name",
                address="123 Main St",
                city="San Juan",
                postal_code="00901",
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            business = BusinessRecord(
                legal_name="Completely Different Name",
                registration_number="12345",
                registration_index="12345-111",
                business_address="456 Other St, San Juan, 00901",  # Same postal code
                status="ACTIVE",
                resident_agent_name="John Doe",
                resident_agent_address="789 Agent St"
            )
            
            restaurant_no_postal = RestaurantRecord(
                name="Different Name",
                address="123 Main St",
                city="San Juan",
                postal_code="00902",  # Different postal
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            name_score_no_postal = matcher._calculate_name_similarity(restaurant_no_postal.name, business.legal_name)
            score_no_postal, _, _, _ = matcher._calculate_match_score(restaurant_no_postal, business, name_score_no_postal)
            
            name_score_with_postal = matcher._calculate_name_similarity(restaurant.name, business.legal_name)
            score_with_postal, _, _, _ = matcher._calculate_match_score(restaurant, business, name_score_with_postal)
            
            # Difference should be approximately 30 (postal code bonus)
            bonus_contribution = score_with_postal - score_no_postal
            assert bonus_contribution >= 25, f"Postal code bonus should add ~30, got {bonus_contribution}"
            assert bonus_contribution <= 35, f"Postal code bonus should add ~30, got {bonus_contribution}"
    
    @pytest.mark.asyncio
    async def test_city_bonus(self):
        """Test city match bonus adds 20 points"""
        async with AsyncMockIncorporationSearcher() as searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=10)
            
            restaurant = RestaurantRecord(
                name="Test Restaurant",
                address="123 Main St",
                city="San Juan",
                postal_code="00901",
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            business = BusinessRecord(
                legal_name="Test Restaurant",
                registration_number="12345",
                registration_index="12345-111",
                business_address="456 Other St, San Juan, 00902",  # Different postal, same city
                status="ACTIVE",
                resident_agent_name="John Doe",
                resident_agent_address="789 Agent St"
            )
            
            name_score = matcher._calculate_name_similarity(restaurant.name, business.legal_name)
            score_with_city, _, _, _ = matcher._calculate_match_score(restaurant, business, name_score)
            
            # With good name match (let's say ~90%) = 45, plus city bonus 20 = ~65
            # Allow some variance for name matching
            assert score_with_city >= 60, f"Expected score with city bonus >= 60, got {score_with_city}"
    
    @pytest.mark.asyncio
    async def test_score_bounds(self):
        """Test that scores are always between 0 and 100"""
        async with AsyncMockIncorporationSearcher() as searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=10)
            
            restaurant = RestaurantRecord(
                name="Completely Different Name",
                address="123 Main St",
                city="San Juan",
                postal_code="00901",
                coordinates=(-66.1, 18.5),
                rating=4.5
            )
            
            business = BusinessRecord(
                legal_name="Totally Unrelated Business Name",
                registration_number="12345",
                registration_index="12345-111",
                business_address="999 Far Away St, Ponce, 00717",
                status="ACTIVE",
                resident_agent_name="John Doe",
                resident_agent_address="789 Agent St"
            )
            
            name_score = matcher._calculate_name_similarity(restaurant.name, business.legal_name)
            score, _, _, _ = matcher._calculate_match_score(restaurant, business, name_score)
            
            assert 0 <= score <= 100, f"Score should be between 0 and 100, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
