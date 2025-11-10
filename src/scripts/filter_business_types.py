#!/usr/bin/env python3
"""
Filter businesses by type for DoorDash compatibility.

This script filters cleaned_restaurants.csv to remove unsupported business types,
creating a DoorDash-compatible dataset.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.business_type_filter import BusinessTypeFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """Filter businesses by type."""
    # Default paths
    input_file = "data/processed/cleaned_restaurants.csv"
    output_file = "data/processed/doordash_filtered_restaurants.csv"
    removed_file = "data/processed/removed_businesses.csv"
    exclusion_list = "src/misc/excluded_business_types.txt"
    inclusion_list = "src/misc/included_business_types.txt"
    
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info(f"Please ensure {input_file} exists before running this script")
        sys.exit(1)
    
    # Validate exclusion list exists
    exclusion_path = Path(exclusion_list)
    if not exclusion_path.exists():
        logger.error(f"Exclusion list not found: {exclusion_list}")
        logger.info("Please ensure the exclusion list file exists")
        sys.exit(1)
    
    # Validate inclusion list exists
    inclusion_path = Path(inclusion_list)
    if not inclusion_path.exists():
        logger.error(f"Inclusion list not found: {inclusion_list}")
        logger.info("Please ensure the inclusion list file exists")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("BUSINESS TYPE FILTERING")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Exclusion list: {exclusion_list}")
    logger.info(f"Inclusion list: {inclusion_list}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Removed records: {removed_file}")
    logger.info("=" * 80)
    logger.info("Logic: Remove if matches exclusion type AND does NOT match any inclusion type")
    logger.info("=" * 80)
    
    try:
        # Create filter with both exclusion and inclusion lists
        filter_obj = BusinessTypeFilter(
            exclusion_list_file=exclusion_list,
            inclusion_list_file=inclusion_list
        )
        
        # Filter the file
        result = filter_obj.filter_file(
            input_file=input_file,
            output_file=output_file,
            removed_file=removed_file
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("FILTERING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Filtering completed successfully")
        logger.info(f"📊 Original records: {result.total_original:,}")
        logger.info(f"✅ Filtered (kept): {result.total_filtered:,}")
        logger.info(f"❌ Removed: {result.total_removed:,}")
        logger.info(f"📈 Removal rate: {(result.total_removed / result.total_original * 100):.1f}%")
        logger.info(f"\n📁 Output file: {output_file}")
        logger.info(f"📁 Removed records: {removed_file}")
        logger.info("=" * 80)
        
        logger.info("\n✅ Next step: Run the pipeline with:")
        logger.info(f"   python main.py --input {output_file}")
        
    except Exception as e:
        logger.error(f"Error during filtering: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

