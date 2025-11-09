"""
Data loading utilities for matcher
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Protocol
from src.models.models import RestaurantRecord, create_restaurant_from_csv_row

logger = logging.getLogger(__name__)


class RestaurantLoader(Protocol):
    """
    Protocol defining the interface for restaurant loaders.
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
    loader = CSVRestaurantLoader()
    return loader.load(csv_path, limit=limit)
