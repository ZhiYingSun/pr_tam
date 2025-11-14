"""
Final customer-facing report generator for Puerto Rico Restaurant Matcher.

Transforms intermediate match data into final customer-facing CSV reports.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generator for final customer-facing reports from matched restaurant data."""
    
    @staticmethod
    def create_incorporation_link(registration_index: str) -> str:
        """Create incorporation document link from registration index."""
        if not registration_index or pd.isna(registration_index):
            return ""
        return f"https://rcp.estado.pr.gov/en/entity-information?c={registration_index}"
    
    def run(
        self,
        output_csv_path: str,
        validation_csv_path: Optional[str] = None
    ) -> Dict:
        """
        Run the transformation pipeline.
        
        Args:
            output_csv_path: Path to save the transformed data
            validation_csv_path: Path to accepted matches CSV
            
        Returns:
            Dictionary with transformation results
        """
        try:
            # Use accepted matches file as primary source
            if not validation_csv_path or not Path(validation_csv_path).exists():
                logger.warning("No accepted matches file provided - cannot generate final report")
                return {
                    'success': False,
                    'error': 'Accepted matches file required',
                    'output_file': None
                }
            
            logger.info(f"Loading accepted matches from {validation_csv_path}")
            df = pd.read_csv(validation_csv_path)
            logger.info(f"Loaded {len(df)} accepted matches")
            
            if df.empty:
                logger.warning("No accepted matches to transform")
                return {
                    'success': False,
                    'error': 'No accepted matches',
                    'output_file': None
                }
            
            # Transform data directly from accepted matches CSV (no joins needed)
            transformed_data = []
            
            for _, row in df.iterrows():
                state = "Puerto Rico"
                incorporation_link = self.create_incorporation_link(row.get('business_registration_index', ''))
                
                transformed_row = {
                    'Location Name': row['restaurant_name'],
                    'Address': row.get('restaurant_address', '') if pd.notna(row.get('restaurant_address')) else '',
                    'City': row.get('restaurant_city', '') if pd.notna(row.get('restaurant_city')) else '',
                    'State': state,
                    'Website': row.get('restaurant_website', '') if pd.notna(row.get('restaurant_website')) else '',
                    'Phone': row.get('restaurant_phone', '') if pd.notna(row.get('restaurant_phone')) else '',
                    'Review Rating': row.get('restaurant_rating', 0) if pd.notna(row.get('restaurant_rating')) else 0,
                    'Number of Reviews': row.get('restaurant_reviews_count', 0) if pd.notna(row.get('restaurant_reviews_count')) else 0,
                    'Primary Business Type': row.get('restaurant_main_type', '') if pd.notna(row.get('restaurant_main_type')) else '',
                    'Legal Name': row['business_legal_name'],
                    'Incorporation Document Link': incorporation_link
                }
                
                transformed_data.append(transformed_row)
            
            # Create output DataFrame
            output_df = pd.DataFrame(transformed_data)
            
            # Save to CSV
            output_path = Path(output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(output_df)} transformed records to {output_path}")
            
            # Print sample of transformed data
            logger.info("\nSample of transformed data:")
            logger.info(output_df.head(3).to_string(index=False))
            
            return {
                'success': True,
                'output_file': str(output_path),
                'record_count': len(output_df)
            }
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'output_file': None
            }

