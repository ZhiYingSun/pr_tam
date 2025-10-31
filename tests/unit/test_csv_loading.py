"""
Unit tests for CSV loading and data model creation
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.utils.loader import load_restaurants
from src.data.models import RestaurantRecord, create_restaurant_from_csv_row


class TestCSVLoading:
    """Test loading restaurants from CSV files"""
    
    def test_load_simple_csv(self):
        """Test loading a simple CSV with valid restaurant data"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Full address,City,Postal code,Longitude,Latitude,Reviews rating,Google ID,Phone,Website,Reviews count,Main type\n")

            f.write('Test Restaurant,123 Main St,San Juan,"00901",-66.1,18.5,4.5,test_id,787-123-4567,https://test.com,100,Restaurant\n')
            csv_path = f.name
        
        try:
            restaurants = load_restaurants(csv_path)
            
            assert len(restaurants) == 1
            assert isinstance(restaurants[0], RestaurantRecord)
            assert restaurants[0].name == "Test Restaurant"
            assert restaurants[0].city == "San Juan"

            # TODO fix the bug where panda strips leading zeros
            assert restaurants[0].postal_code == "00901", f"Unexpected postal code: {restaurants[0].postal_code}"
            assert restaurants[0].coordinates == (-66.1, 18.5)
            assert restaurants[0].rating == 4.5
            assert restaurants[0].google_id == "test_id"
            assert restaurants[0].phone == "787-123-4567"
        finally:
            Path(csv_path).unlink()
    
    def test_load_csv_with_limit(self):
        """Test loading CSV with a limit parameter"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Full address,City,Postal code,Longitude,Latitude,Reviews rating\n")
            for i in range(5):
                f.write(f'Restaurant {i},123 Main St,San Juan,00901,-66.1,18.5,4.5\n')
            csv_path = f.name
        
        try:
            restaurants = load_restaurants(csv_path, limit=3)
            assert len(restaurants) == 3
        finally:
            Path(csv_path).unlink()
    
    def test_load_csv_missing_optional_fields(self):
        """Test loading CSV when optional fields are missing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Full address,City,Postal code,Longitude,Latitude,Reviews rating\n")
            f.write('Test Restaurant,123 Main St,San Juan,00901,-66.1,18.5,4.5\n')
            csv_path = f.name
        
        try:
            restaurants = load_restaurants(csv_path)
            
            assert len(restaurants) == 1
            # Optional fields should be None
            assert restaurants[0].google_id is None or restaurants[0].google_id == ""
            assert restaurants[0].phone is None or restaurants[0].phone == ""
        finally:
            Path(csv_path).unlink()
    
    def test_load_csv_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_restaurants("/nonexistent/path/to/file.csv")
    
    def test_load_empty_csv(self):
        """Test that loading an empty CSV raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Full address,City,Postal code\n")  # Header only
            csv_path = f.name
        
        try:
            with pytest.raises(ValueError, match="empty"):
                load_restaurants(csv_path)
        finally:
            Path(csv_path).unlink()


class TestDataModelCreation:
    """Test creating RestaurantRecord from CSV row"""
    
    def test_create_restaurant_from_complete_row(self):
        """Test creating RestaurantRecord from a complete CSV row"""
        row = {
            'Name': 'Test Restaurant',
            'Full address': '123 Main St',
            'City': 'San Juan',
            'Postal code': '00901',
            'Longitude': -66.1,
            'Latitude': 18.5,
            'Reviews rating': 4.5,
            'Google ID': 'test_id',
            'Phone': '787-123-4567',
            'Website': 'https://test.com',
            'Reviews count': 100,
            'Main type': 'Restaurant'
        }
        
        restaurant = create_restaurant_from_csv_row(row)
        
        assert isinstance(restaurant, RestaurantRecord)
        assert restaurant.name == "Test Restaurant"
        assert restaurant.address == "123 Main St"
        assert restaurant.city == "San Juan"
        assert restaurant.postal_code == "00901"
        assert restaurant.coordinates == (-66.1, 18.5)
        assert restaurant.rating == 4.5
        assert restaurant.google_id == "test_id"
        assert restaurant.phone == "787-123-4567"
        assert restaurant.website == "https://test.com"
        assert restaurant.reviews_count == 100
        assert restaurant.main_type == "Restaurant"
    
    def test_create_restaurant_from_minimal_row(self):
        """Test creating RestaurantRecord from minimal CSV row (only required fields)"""
        row = {
            'Name': 'Minimal Restaurant',
            'Full address': '123 Main St',
            'City': 'San Juan',
            'Postal code': '00901',
            'Longitude': -66.1,
            'Latitude': 18.5,
            'Reviews rating': 4.0
        }
        
        restaurant = create_restaurant_from_csv_row(row)
        
        assert isinstance(restaurant, RestaurantRecord)
        assert restaurant.name == "Minimal Restaurant"

        # Optional fields should handle missing values gracefully
        assert restaurant.google_id is None or restaurant.google_id == ""
        assert restaurant.phone is None or restaurant.phone == ""
    
    def test_create_restaurant_coordinates_as_strings(self):
        """Test handling coordinates that come as strings from CSV"""
        row = {
            'Name': 'Test Restaurant',
            'Full address': '123 Main St',
            'City': 'San Juan',
            'Postal code': '00901',
            'Longitude': '-66.1',
            'Latitude': '18.5',
            'Reviews rating': 4.5
        }
        
        restaurant = create_restaurant_from_csv_row(row)

        assert isinstance(restaurant.coordinates[0], float)
        assert isinstance(restaurant.coordinates[1], float)
        assert restaurant.coordinates == (-66.1, 18.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

