"""
Data loading utilities for matcher
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Protocol
from src.data.models import RestaurantRecord, create_restaurant_from_csv_row

logger = logging.getLogger(__name__)


class RestaurantLoader(Protocol):
    """
    Protocol defining the interface for restaurant loaders.
    Allows dependency injection and easy testing.
    """
    def load(self, source: str, limit: Optional[int] = None) -> List[RestaurantRecord]:
        """
        Load restaurants from a source.
        
        Args:
            source: Source identifier (e.g., file path, database connection string)
            limit: Optional limit on number of restaurants to load
            
        Returns:
            List of RestaurantRecord objects
            
        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If source is malformed or empty
        """
        ...


class CSVRestaurantLoader:
    """
    Loader implementation that loads restaurants from CSV files.
    """
    
    def load(self, csv_path: str, limit: Optional[int] = None) -> List[RestaurantRecord]:
        """
        Load restaurants from CSV file.
        
        Args:
            csv_path: Path to the cleaned restaurants CSV file
            limit: Optional limit on number of restaurants to load (for testing)
            
        Returns:
            List of RestaurantRecord objects
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is malformed or empty
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading restaurants from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            logger.info(f"Loaded {len(df)} restaurants from CSV")

            if limit:
                df = df.head(limit)
                logger.info(f"Limited to {len(df)} restaurants for processing")

            # Convert each row to RestaurantRecord
            restaurants = []
            failed_conversions = 0
            
            for index, row in df.iterrows():
                try:
                    restaurant = create_restaurant_from_csv_row(row.to_dict())
                    restaurants.append(restaurant)
                except Exception as e:
                    failed_conversions += 1
                    logger.warning(f"Failed to convert row {index}: {e}")
                    continue
            
            logger.info(f"Successfully converted {len(restaurants)} restaurants")
            if failed_conversions > 0:
                logger.warning(f"Failed to convert {failed_conversions} restaurants")
            
            return restaurants
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or malformed")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading restaurants: {e}")
            raise


def load_restaurants(csv_path: str, limit: Optional[int] = None) -> List[RestaurantRecord]:
    """
    Convenience function to load restaurants from CSV file.
    Uses CSVRestaurantLoader internally for backward compatibility.
    
    Args:
        csv_path: Path to the cleaned restaurants CSV file
        limit: Optional limit on number of restaurants to load (for testing)
        
    Returns:
        List of RestaurantRecord objects
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV file is malformed or empty
    """
    loader = CSVRestaurantLoader()
    return loader.load(csv_path, limit=limit)


def load_restaurants_batch(csv_path: str, batch_size: int = 1000) -> List[List[RestaurantRecord]]:
    """
    Load restaurants in batches for memory-efficient processing.
    
    Args:
        csv_path: Path to the cleaned restaurants CSV file
        batch_size: Number of restaurants per batch
        
    Returns:
        List of batches, each containing RestaurantRecord objects
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Loading restaurants in batches of {batch_size} from {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        total_restaurants = len(df)
        batches = []
        
        # Process in batches
        for start_idx in range(0, total_restaurants, batch_size):
            end_idx = min(start_idx + batch_size, total_restaurants)
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_restaurants = []
            failed_conversions = 0
            
            for index, row in batch_df.iterrows():
                try:
                    restaurant = create_restaurant_from_csv_row(row.to_dict())
                    batch_restaurants.append(restaurant)
                except Exception as e:
                    failed_conversions += 1
                    logger.warning(f"Failed to convert row {index}: {e}")
                    continue
            
            batches.append(batch_restaurants)
            logger.info(f"Processed batch {len(batches)}: {len(batch_restaurants)} restaurants")
        
        logger.info(f"Created {len(batches)} batches with {total_restaurants} total restaurants")
        return batches
        
    except Exception as e:
        logger.error(f"Error loading restaurants in batches: {e}")
        raise


def get_data_summary(csv_path: str) -> dict:
    """
    Get summary statistics about the restaurant data.
    
    Args:
        csv_path: Path to the cleaned restaurants CSV file
        
    Returns:
        Dictionary with summary statistics
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        summary = {
            'total_restaurants': len(df),
            'cities': df['City'].nunique(),
            'postal_codes': df['Postal code'].nunique(),
            'main_types': df['Main type'].nunique(),
            'avg_rating': df['Reviews rating'].mean(),
            'total_reviews': df['Reviews count'].sum(),
            'top_cities': df['City'].value_counts().head(10).to_dict(),
            'top_types': df['Main type'].value_counts().head(10).to_dict(),
            'rating_distribution': {
                'min': df['Reviews rating'].min(),
                'max': df['Reviews rating'].max(),
                'median': df['Reviews rating'].median()
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise


def validate_restaurant_data(restaurants: List[RestaurantRecord]) -> dict:
    """
    Validate restaurant data and return validation results.
    
    Args:
        restaurants: List of RestaurantRecord objects
        
    Returns:
        Dictionary with validation results
    """
    total_count = len(restaurants)
    validation_results = {
        'total_restaurants': total_count,
        'valid_restaurants': 0,
        'invalid_restaurants': 0,
        'missing_name': 0,
        'missing_address': 0,
        'missing_city': 0,
        'missing_postal_code': 0,
        'invalid_coordinates': 0,
        'invalid_rating': 0,
        'issues': []
    }
    
    for i, restaurant in enumerate(restaurants):
        is_valid = True
        restaurant_issues = []
        
        # Check required fields
        if not restaurant.name or restaurant.name.strip() == '':
            validation_results['missing_name'] += 1
            restaurant_issues.append('missing_name')
            is_valid = False
        
        if not restaurant.address or restaurant.address.strip() == '':
            validation_results['missing_address'] += 1
            restaurant_issues.append('missing_address')
            is_valid = False
        
        if not restaurant.city or restaurant.city.strip() == '':
            validation_results['missing_city'] += 1
            restaurant_issues.append('missing_city')
            is_valid = False
        
        if not restaurant.postal_code or restaurant.postal_code.strip() == '':
            validation_results['missing_postal_code'] += 1
            restaurant_issues.append('missing_postal_code')
            is_valid = False
        
        # Check coordinates
        if restaurant.coordinates[0] == 0 and restaurant.coordinates[1] == 0:
            validation_results['invalid_coordinates'] += 1
            restaurant_issues.append('invalid_coordinates')
            is_valid = False
        
        # Check rating
        if restaurant.rating < 0 or restaurant.rating > 5:
            validation_results['invalid_rating'] += 1
            restaurant_issues.append('invalid_rating')
            is_valid = False
        
        if is_valid:
            validation_results['valid_restaurants'] += 1
        else:
            validation_results['invalid_restaurants'] += 1
            validation_results['issues'].append({
                'index': i,
                'name': restaurant.name,
                'issues': restaurant_issues
            })
    
    return validation_results


def save_restaurants_to_csv(restaurants: List[RestaurantRecord], output_path: str) -> None:
    """
    Save RestaurantRecord objects to CSV file.
    
    Args:
        restaurants: List of RestaurantRecord objects
        output_path: Path to save the CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    data = []
    for restaurant in restaurants:
        data.append({
            'name': restaurant.name,
            'address': restaurant.address,
            'city': restaurant.city,
            'postal_code': restaurant.postal_code,
            'longitude': restaurant.coordinates[0],
            'latitude': restaurant.coordinates[1],
            'rating': restaurant.rating,
            'google_id': restaurant.google_id,
            'phone': restaurant.phone,
            'website': restaurant.website,
            'reviews_count': restaurant.reviews_count,
            'main_type': restaurant.main_type
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(restaurants)} restaurants to {output_path}")


if __name__ == "__main__":
    # Test the data loading functions
    csv_path = "data/processed/cleaned_restaurants.csv"
    
    try:
        # Test basic loading
        print("Testing basic data loading...")
        restaurants = load_restaurants(csv_path, limit=10)
        print(f"Loaded {len(restaurants)} restaurants")
        
        # Test data summary
        print("\nTesting data summary...")
        summary = get_data_summary(csv_path)
        print(f"Total restaurants: {summary['total_restaurants']}")
        print(f"Cities: {summary['cities']}")
        print(f"Average rating: {summary['avg_rating']:.2f}")
        
        # Test validation
        print("\nTesting data validation...")
        validation = validate_restaurant_data(restaurants)
        print(f"Valid restaurants: {validation['valid_restaurants']}")
        print(f"Invalid restaurants: {validation['invalid_restaurants']}")
        
        print("\n Data loading tests completed!")
        
    except Exception as e:
        print(f" Error: {e}")
