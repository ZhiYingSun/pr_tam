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
        validation_csv_path: Optional[str] = None,
        filtered_csv_path: Optional[str] = None
    ) -> Dict:
        """
        Run the transformation pipeline.
        
        Args:
            output_csv_path: Path to save the transformed data
            validation_csv_path: Path to accepted matches CSV
            filtered_csv_path: Path to filtered businesses CSV (includes all restaurants)
            
        Returns:
            Dictionary with transformation results
        """
        try:
            # Load filtered businesses as the base (all restaurants)
            if not filtered_csv_path or not Path(filtered_csv_path).exists():
                logger.warning("No filtered businesses file provided - cannot generate final report")
                return {
                    'success': False,
                    'error': 'Filtered businesses file required',
                    'output_file': None
                }
            
            logger.info(f"Loading filtered businesses from {filtered_csv_path}")
            filtered_df = pd.read_csv(filtered_csv_path)
            logger.info(f"Loaded {len(filtered_df)} filtered businesses")
            
            if filtered_df.empty:
                logger.warning("No filtered businesses to transform")
                return {
                    'success': False,
                    'error': 'No filtered businesses',
                    'output_file': None
                }
            
            # Load accepted matches (optional - will be left-joined)
            matches_df = None
            if validation_csv_path and Path(validation_csv_path).exists():
                logger.info(f"Loading accepted matches from {validation_csv_path}")
                matches_df = pd.read_csv(validation_csv_path)
                logger.info(f"Loaded {len(matches_df)} accepted matches")
            else:
                logger.info("No accepted matches file - will create report with empty incorporation columns")
            
            # Transform data: all filtered businesses with optional incorporation data
            transformed_data = []
            
            for _, row in filtered_df.iterrows():
                restaurant_name = row['Name']
                
                # Find matching incorporation record if exists
                match_row = None
                if matches_df is not None and not matches_df.empty:
                    match = matches_df[matches_df['restaurant_name'] == restaurant_name]
                    if not match.empty:
                        match_row = match.iloc[0]
                
                state = "Puerto Rico"
                
                # Base restaurant data (always present)
                transformed_row = {
                    'Location Name': restaurant_name,
                    'Address': row.get('Full address', '') if pd.notna(row.get('Full address')) else '',
                    'City': row.get('City', '') if pd.notna(row.get('City')) else '',
                    'State': state,
                    'Website': row.get('Website', '') if pd.notna(row.get('Website')) else '',
                    'Phone': row.get('Phone', '') if pd.notna(row.get('Phone')) else '',
                    'Review Rating': row.get('Reviews rating', 0) if pd.notna(row.get('Reviews rating')) else 0,
                    'Number of Reviews': row.get('Reviews count', 0) if pd.notna(row.get('Reviews count')) else 0,
                    'Primary Business Type': row.get('Main type', '') if pd.notna(row.get('Main type')) else '',
                }
                
                # Incorporation data (empty if no match)
                if match_row is not None:
                    # Get values and convert None/NaN to empty string
                    legal_name = match_row.get('business_legal_name', '')
                    if pd.isna(legal_name):
                        legal_name = ''
                    
                    reg_index = match_row.get('business_registration_index', '')
                    incorporation_link = self.create_incorporation_link(reg_index)
                    
                    transformed_row['Legal Name'] = legal_name
                    transformed_row['Incorporation Document Link'] = incorporation_link
                else:
                    transformed_row['Legal Name'] = ''
                    transformed_row['Incorporation Document Link'] = ''
                
                transformed_data.append(transformed_row)
            
            # Create output DataFrame
            output_df = pd.DataFrame(transformed_data)
            
            # Replace NaN with empty strings for cleaner output
            output_df = output_df.fillna('')
            
            # Save to CSV (na_rep='' ensures NaN values are saved as empty strings)
            output_path = Path(output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False, na_rep='')
            logger.info(f"Saved {len(output_df)} transformed records to {output_path}")
            
            # Log summary statistics
            matched_count = len([r for r in transformed_data if r['Legal Name']])
            logger.info(f"Report contains {len(output_df)} total restaurants ({matched_count} with incorporation data)")
            
            # Print sample of transformed data
            logger.info("\nSample of transformed data:")
            logger.info(output_df.head(3).to_string(index=False))
            
            return {
                'success': True,
                'output_file': str(output_path),
                'record_count': len(output_df),
                'matched_count': matched_count
            }
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'output_file': None
            }

