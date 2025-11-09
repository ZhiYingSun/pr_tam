"""
Final customer-facing report generator for Puerto Rico Restaurant Matcher.

Transforms intermediate match data into final customer-facing CSV reports.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class FinalCustomerFacingReportGenerator:
    """Generator for final customer-facing reports from matched restaurant data."""
    
    @staticmethod
    def create_incorporation_link(registration_index: str) -> str:
        """Create incorporation document link from registration index."""
        if not registration_index or pd.isna(registration_index):
            return ""
        return f"https://rcp.estado.pr.gov/en/entity-information?c={registration_index}"
    
    def run(
        self,
        input_csv_path: str,
        output_csv_path: str,
        validation_csv_path: Optional[str] = None
    ) -> Dict:
        """
        Run the transformation pipeline.
        
        Args:
            input_csv_path: Path to the original matched restaurants CSV
            output_csv_path: Path to save the transformed data
            validation_csv_path: Optional path to validation results CSV
            
        Returns:
            Dictionary with transformation results
        """
        logger.info(f"Loading data from {input_csv_path}")
        
        try:
            # Load the original matched data
            df = pd.read_csv(input_csv_path)
            logger.info(f"Loaded {len(df)} matched restaurants")
            
            if df.empty:
                logger.warning("No data to transform")
                return {
                    'success': False,
                    'error': 'Empty input data',
                    'output_file': None
                }
            
            # If validation results are provided, merge them
            if validation_csv_path and Path(validation_csv_path).exists():
                logger.info(f"Loading validation results from {validation_csv_path}")
                validation_df = pd.read_csv(validation_csv_path)
                
                # Merge validation results with original data
                df = df.merge(
                    validation_df[['restaurant_name', 'business_legal_name', 'openai_recommendation']],
                    on=['restaurant_name', 'business_legal_name'],
                    how='left'
                )
                
                # Filter to only include accepted matches if validation data is available
                if 'openai_recommendation' in df.columns:
                    accepted_df = df[df['openai_recommendation'] == 'accept']
                    logger.info(f"Filtered to {len(accepted_df)} accepted matches")
                    df = accepted_df
            
            # Transform data
            transformed_data = []
            
            for _, row in df.iterrows():
                state = "Puerto Rico"
                incorporation_link = self.create_incorporation_link(row.get('business_registration_index', ''))
                
                transformed_row = {
                    'Location Name': row['restaurant_name'],
                    'Address': row['restaurant_address'],
                    'City': row['restaurant_city'],
                    'State': state,
                    'Website': row.get('restaurant_website', '') if pd.notna(row.get('restaurant_website')) else '',
                    'Phone': row.get('restaurant_phone', '') if pd.notna(row.get('restaurant_phone')) else '',
                    'Review Rating': row['restaurant_rating'],
                    'Number of Reviews': row.get('restaurant_reviews_count', 0),
                    'Primary Business Type': row.get('restaurant_main_type', ''),
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

