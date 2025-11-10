"""
Business Type Filter

Filters businesses based on inclusion/exclusion lists loaded from text files.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Set, Optional, List
from dataclasses import dataclass

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of filtering operation."""
    model_config = {"arbitrary_types_allowed": True}
    
    filtered_df: pd.DataFrame
    removed_df: pd.DataFrame
    total_original: int
    total_filtered: int
    total_removed: int
    removal_reasons: dict[str, int]


class BusinessTypeListLoader:
    """
    Loads business type lists from text files.
    """
    
    @staticmethod
    def load_from_file(file_path: str) -> Set[str]:
        """
        Load business types from a text file (one per line).
        
        Args:
            file_path: Path to text file containing business types
            
        Returns:
            Set of business type strings (normalized: stripped, no empty lines)
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Business type list file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            types = {line.strip() for line in f if line.strip()}
        
        logger.info(f"Loaded {len(types)} business types from {file_path}")
        return types


class BusinessTypeFilter:
    """
    Filters businesses based on combined inclusion/exclusion logic.
    
    Logic: Remove a business if it matches an exclusion type AND does NOT match any inclusion type.
    This allows businesses with mixed types (e.g., "Bar" + "Restaurant") to be kept if they
    have at least one inclusion type.
    
    Usage:
        # Using both exclusion and inclusion lists
        filter_obj = BusinessTypeFilter(
            exclusion_list_file="src/misc/excluded_business_types.txt",
            inclusion_list_file="src/misc/included_business_types.txt"
        )
        result = filter_obj.filter(df)
        
        # Injecting lists directly (for testing)
        filter_obj = BusinessTypeFilter(
            excluded_types={"Bar", "Club"},
            included_types={"Restaurant", "Cafe"}
        )
        result = filter_obj.filter(df)
    """
    
    def __init__(
        self,
        exclusion_list_file: Optional[str] = None,
        inclusion_list_file: Optional[str] = None,
        excluded_types: Optional[Set[str]] = None,
        included_types: Optional[Set[str]] = None,
        main_type_column: str = "Main type",
        all_types_column: str = "All types"
    ):
        """
        Initialize business type filter with combined exclusion/inclusion logic.
        
        Args:
            exclusion_list_file: Path to text file with excluded business types (one per line)
            inclusion_list_file: Path to text file with included business types (one per line)
            excluded_types: Set of excluded business types (alternative to file)
            included_types: Set of included business types (alternative to file)
            main_type_column: Name of column containing main business type (default: "Main type")
            all_types_column: Name of column containing all business types (default: "All types")
            
        Note:
            Both exclusion and inclusion lists are required for the combined logic.
        """
        self.main_type_column = main_type_column
        self.all_types_column = all_types_column
        
        # Load exclusion list
        if exclusion_list_file:
            self.excluded_types = BusinessTypeListLoader.load_from_file(exclusion_list_file)
        elif excluded_types:
            self.excluded_types = excluded_types
        else:
            raise ValueError("Must provide exclusion_list_file or excluded_types")
        
        # Load inclusion list
        if inclusion_list_file:
            self.included_types = BusinessTypeListLoader.load_from_file(inclusion_list_file)
        elif included_types:
            self.included_types = included_types
        else:
            raise ValueError("Must provide inclusion_list_file or included_types")
        
        logger.info(
            f"Initialized BusinessTypeFilter with combined exclusion/inclusion logic: "
            f"{len(self.excluded_types)} excluded types, {len(self.included_types)} included types"
        )
    
    def _parse_all_types(self, all_types_str: str) -> Set[str]:
        """
        Parse comma-separated business types string into a set.
        
        Args:
            all_types_str: Comma-separated string of business types
            
        Returns:
            Set of normalized business type strings
        """
        if pd.isna(all_types_str) or not all_types_str:
            return set()
        return {t.strip() for t in str(all_types_str).split(',') if t.strip()}
    
    def _get_all_business_types(self, row: pd.Series) -> Set[str]:
        """
        Get all business types from a row (Main type + All types).
        
        Args:
            row: DataFrame row containing business record
            
        Returns:
            Set of all business types for this record
        """
        types = set()
        
        # Add main type
        main_type = row.get(self.main_type_column)
        if pd.notna(main_type) and main_type:
            types.add(str(main_type).strip())
        
        # Add all types from "All types" column
        all_types_str = row.get(self.all_types_column)
        if pd.notna(all_types_str) and all_types_str:
            types.update(self._parse_all_types(str(all_types_str)))
        
        return types
    
    def _should_include(self, row: pd.Series) -> bool:
        """
        Determine if a business record should be included in the filtered result.
        
        Logic: Remove if (matches exclusion type) AND (does NOT match any inclusion type).
        This keeps businesses that have at least one inclusion type, even if they also
        have exclusion types (e.g., "Bar" + "Restaurant" = keep).
        
        Args:
            row: DataFrame row containing business record
            
        Returns:
            True if should be included, False if should be excluded
        """
        all_types = self._get_all_business_types(row)
        
        # Check if any type matches exclusion list
        matches_exclusion = any(biz_type in self.excluded_types for biz_type in all_types)
        
        # Check if any type matches inclusion list
        matches_inclusion = any(biz_type in self.included_types for biz_type in all_types)
        
        # Remove if: matches exclusion AND does NOT match inclusion
        # Keep if: doesn't match exclusion OR matches inclusion
        return not matches_exclusion or matches_inclusion
    
    def filter(self, df: pd.DataFrame) -> FilterResult:
        """
        Filter DataFrame based on combined exclusion/inclusion logic.
        
        Logic: Remove if (matches exclusion type) AND (does NOT match any inclusion type).
        
        Args:
            df: Input DataFrame with business records
            
        Returns:
            FilterResult containing filtered DataFrame, removed records, and statistics
        """
        original_count = len(df)
        logger.info(f"Filtering {original_count:,} businesses by business type")
        
        # Check if required columns exist
        if self.main_type_column not in df.columns:
            raise ValueError(
                f"Main type column '{self.main_type_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        if self.all_types_column not in df.columns:
            raise ValueError(
                f"All types column '{self.all_types_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        df_copy['_removal_reason'] = None
        
        # Filter based on combined exclusion/inclusion logic
        mask = df_copy.apply(self._should_include, axis=1)
        
        # Mark removed records
        removed_mask = ~mask
        df_copy.loc[removed_mask, '_removal_reason'] = 'Matches exclusion type but no inclusion type'
        
        # Separate filtered and removed
        filtered_df = df_copy[mask].copy()
        removed_df = df_copy[removed_mask].copy()
        
        # Remove internal column
        filtered_df = filtered_df.drop(columns=['_removal_reason'], errors='ignore')
        if '_removal_reason' in removed_df.columns:
            removed_df = removed_df.rename(columns={'_removal_reason': 'Removal reason'})
        
        filtered_count = len(filtered_df)
        removed_count = len(removed_df)
        
        # Count removals by main business type
        removal_reasons = {}
        if len(removed_df) > 0:
            type_counts = removed_df[self.main_type_column].value_counts()
            removal_reasons['Matches exclusion type but no inclusion type'] = removed_count
            logger.info(f"Removed {removed_count:,} businesses by type:")
            for biz_type, count in type_counts.head(10).items():
                logger.info(f"  {biz_type}: {count:,}")
            if len(type_counts) > 10:
                logger.info(f"  ... and {len(type_counts) - 10} more types")
        
        logger.info(
            f"Filtered {original_count:,} -> {filtered_count:,} businesses "
            f"({removed_count:,} removed)"
        )
        
        return FilterResult(
            filtered_df=filtered_df,
            removed_df=removed_df,
            total_original=original_count,
            total_filtered=filtered_count,
            total_removed=removed_count,
            removal_reasons=removal_reasons
        )
    
    def filter_file(
        self,
        input_file: str,
        output_file: str,
        removed_file: Optional[str] = None
    ) -> FilterResult:
        """
        Filter a CSV file and save results.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save filtered CSV file
            removed_file: Optional path to save removed records CSV file
            
        Returns:
            FilterResult containing filtering statistics
        """
        logger.info(f"Loading businesses from {input_file}")
        df = pd.read_csv(input_file)
        
        result = self.filter(df)
        
        # Save filtered data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.filtered_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(result.filtered_df):,} filtered businesses to {output_file}")
        
        # Save removed data if requested
        if removed_file and len(result.removed_df) > 0:
            removed_path = Path(removed_file)
            removed_path.parent.mkdir(parents=True, exist_ok=True)
            result.removed_df.to_csv(removed_file, index=False)
            logger.info(f"Saved {len(result.removed_df):,} removed businesses to {removed_file}")
        
        return result

