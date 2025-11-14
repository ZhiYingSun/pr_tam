import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any, Coroutine
from datetime import datetime, timedelta

from src.utils.loader import RestaurantLoader, CSVRestaurantLoader
from src.utils.intermediate_match_csv_generator import generate_all_outputs, print_match_statistics
from src.utils.business_filter import BusinessFilter
from src.models.models import MatchingConfig, RestaurantRecord, MatchResult, PipelineResult
from src.models.validation_models import ValidationResult
from src.matchers.matcher import RestaurantMatcher
from src.searchers.searcher import IncorporationSearcher
from src.validators.llm_validator import LLMValidator
from src.utils.report_generator import ReportGenerator
from src.clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the complete pipeline execution."""
    
    def __init__(
        self,
        openai_client: OpenAIClient,
        searcher: IncorporationSearcher,
        config: MatchingConfig,
        loader: RestaurantLoader,
        report_generator: ReportGenerator
    ):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            openai_client: OpenAIClient instance
            searcher: IncorporationSearcher instance
            config: Matching configuration
            loader: RestaurantLoader instance
            transformation_pipeline: ReportGenerator instance
        """
        self.config = config
        self.searcher = searcher
        self.loader = loader
        self.report_generator = report_generator
        self.openai_client = openai_client

        self.validator = LLMValidator(
            openai_client=openai_client,
            model="gpt-4o-mini"
        )

    async def process_restaurant(
        self,
        restaurant: RestaurantRecord,
        matcher: RestaurantMatcher
    ) -> Tuple[Optional[MatchResult], Optional[ValidationResult]]:
        """
        Process a single restaurant through the full match → validate pipeline.

        Args:
        restaurant: RestaurantRecord to process.
        matcher: Shared RestaurantMatcher instance.

        Returns:
        Tuple of (MatchResult, ValidationResult). 
        - MatchResult: Selected best match from candidates, or None if no match found
        - ValidationResult: LLM validation result with selected best match, or None if validation failed
        """
        # Step 1: Find top 3 candidate matches
        match_results = []
        try:
            match_results = await matcher.find_best_match(restaurant)
        except Exception as e:
            logger.error(f"Error matching restaurant '{restaurant.name}': {e}", exc_info=True)

        # Step 2: Validate all candidates and select the best one
        validation_result = None
        selected_match = None
        
        if match_results:
            try:
                selected_match, validation_result = await self.validator.validate_best_match_from_candidates(match_results)
                return selected_match, validation_result
            except Exception as e:
                logger.error(f"Error validating matches for '{restaurant.name}': {e}", exc_info=True)

        return selected_match, validation_result
    
    async def run(
        self,
        input_csv: str,
        output_dir: str = "data/output",
        limit: Optional[int] = None,
        exclusion_list: Optional[str] = None,
        inclusion_list: Optional[str] = None
    ) -> PipelineResult:
        timestamp, output_path, start_time = self._initialize_pipeline_run(
            input_csv, output_dir, limit
        )
        
        # Apply business type filtering
        filtered_csv = self._apply_business_filter(
                input_csv, output_path, timestamp, exclusion_list, inclusion_list
        )
        
        # Load restaurants
        logger.info("Loading restaurants...")
        restaurants = self.loader.load(filtered_csv, limit=limit)
        logger.info(f"Loaded {len(restaurants)} restaurants")

        async with self.searcher:
            matcher = RestaurantMatcher(self.searcher, self.openai_client)

            logger.info(f"Processing {len(restaurants)} restaurants concurrently...")
            tasks = [self.process_restaurant(restaurant, matcher) for restaurant in restaurants]
            restaurant_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        match_results, validation_results, error_count, error_rate = self._process_restaurant_results(
            restaurant_results, restaurants
        )
        
        # Generate output files
        output_files = generate_all_outputs(match_results, str(output_path))
        
        # Calculate and print statistics
        statistics = print_match_statistics(match_results)

        # Save validation results if available
        if validation_results:
            validation_file = self._save_validation_results(validation_results, output_path, timestamp)
        else:
            validation_file = None
        
        # Transformation - always run to include all filtered businesses
        final_output = self._run_transformation(
            output_path, timestamp, validation_file, filtered_csv
        )
        
        duration = self._log_pipeline_completion(start_time)
        
        return PipelineResult(
            success=error_rate < 0.5,  # Success only if error rate below threshold
            error_count=error_count,
            error_rate=error_rate,
            timestamp=timestamp,
            input_csv=input_csv,
            output_dir=str(output_path),
            results=match_results,
            validation_results=validation_results,
            statistics=statistics,
            output_files=output_files,
            matched_file=output_files.matched_csv if match_results else None,
            validation_file=validation_file,
            final_output=final_output,
            duration=duration
        )
    
    def _process_restaurant_results(
        self,
        restaurant_results: List[Union[Tuple[Optional[MatchResult], Optional[ValidationResult]], Exception]],
        restaurants: List[RestaurantRecord]
    ) -> Tuple[List[MatchResult], List[ValidationResult], int, float]:
        match_results = []
        validation_results = []
        error_count = 0
        
        for i, result in enumerate(restaurant_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing restaurant '{restaurants[i].name}': {result}")
                error_count += 1
                continue
            
            match_result, validation_result = result
            
            # Add the match result if it exists
            if match_result:
                match_results.append(match_result)
            
            if validation_result:
                validation_results.append(validation_result)
        
        # Calculate error rate and check threshold
        error_rate = error_count / len(restaurants) if restaurants else 0.0
        if error_rate > 0.5:
            raise RuntimeError(
                f"Pipeline failure: {error_count}/{len(restaurants)} restaurants failed "
                f"({error_rate*100:.1f}% error rate exceeds 50% threshold)"
            )
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing ({error_rate*100:.1f}% error rate)")
        
        logger.info(f" Processed {len(restaurants)} restaurants")
        logger.info(f"   Matches found: {len(match_results)}")
        logger.info(f"   Validated: {len(validation_results)}")
        
        return match_results, validation_results, error_count, error_rate
    
    def _log_pipeline_completion(self, start_time: datetime) -> timedelta:
        """
        Log pipeline completion with duration.
        
        Args:
            start_time: Pipeline start time
            
        Returns:
            Duration as timedelta
        """
        duration = datetime.now() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 80)
        return duration
    
    def _initialize_pipeline_run(
        self,
        input_csv: str,
        output_dir: str,
        limit: Optional[int]
    ) -> Tuple[str, Path, datetime]:
        """
        Initialize pipeline run: setup paths, timestamps, and log configuration.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Output directory for results
            limit: Optional limit on number of restaurants
        Returns:
            Tuple of (timestamp, output_path, start_time)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Input: {input_csv}")
        logger.info(f"  Output: {output_dir}")
        if limit:
            logger.info(f"  Limit: {limit} restaurants")
        logger.info(f"  Validation: enabled")
        logger.info(f"  Transformation: enabled")
        logger.info("=" * 80)
        
        return timestamp, output_path, start_time
    
    def _apply_business_filter(
        self,
        input_csv: str,
        output_path: Path,
        timestamp: str,
        exclusion_list: Optional[str],
        inclusion_list: Optional[str]
    ) -> str:
        """
        Apply business type filtering to the input CSV.
        
        Args:
            input_csv: Path to input CSV file
            output_path: Output directory path
            timestamp: Timestamp for file naming
            exclusion_list: Path to exclusion list file
            inclusion_list: Path to inclusion list file
            
        Returns:
            Path to filtered CSV file
        """
        logger.info("=" * 80)
        logger.info("APPLYING BUSINESS FILTERING")
        logger.info("=" * 80)
        
        if not exclusion_list or not inclusion_list:
            raise ValueError("Both exclusion_list and inclusion_list are required for filtering")
        
        # Validate filter list files exist
        exclusion_path = Path(exclusion_list)
        inclusion_path = Path(inclusion_list)
        
        if not exclusion_path.exists():
            raise FileNotFoundError(f"Exclusion list not found: {exclusion_list}")
        if not inclusion_path.exists():
            raise FileNotFoundError(f"Inclusion list not found: {inclusion_list}")
        
        logger.info(f"Exclusion list: {exclusion_list}")
        logger.info(f"Inclusion list: {inclusion_list}")
        logger.info("Filtering logic:")
        logger.info("  1. Remove closed businesses (Is closed = Yes)")
        logger.info("  2. Remove if matches exclusion type AND does NOT match any inclusion type")
        logger.info("-" * 80)
        
        try:
            # Create filter with both exclusion and inclusion lists
            filter_obj = BusinessFilter(
                exclusion_list_file=exclusion_list,
                inclusion_list_file=inclusion_list
            )
            
            # Create filtered output path
            filtered_dir = output_path / "filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            filtered_csv = filtered_dir / f"filtered_businesses_{timestamp}.csv"
            removed_csv = filtered_dir / f"removed_businesses_{timestamp}.csv"
            
            # Filter the file
            result = filter_obj.filter_file(
                input_file=input_csv,
                output_file=str(filtered_csv),
                removed_file=str(removed_csv)
            )
            
            # Log summary
            logger.info("-" * 80)
            logger.info("FILTERING SUMMARY")
            logger.info("-" * 80)
            logger.info(f"Original records: {result.total_original:,}")
            logger.info(f"Filtered (kept): {result.total_filtered:,}")
            logger.info(f"Removed: {result.total_removed:,}")
            logger.info(f"Removal rate: {(result.total_removed / result.total_original * 100):.1f}%")
            logger.info(f"Filtered CSV: {filtered_csv}")
            logger.info(f"Removed CSV: {removed_csv}")
            logger.info("=" * 80)
            
            return str(filtered_csv)
            
        except Exception as e:
            logger.error(f"Error during business type filtering: {e}", exc_info=True)
            raise
    
    def _save_validation_results(
        self,
        validation_results: List[ValidationResult],
        output_path: Path,
        timestamp: str
    ) -> Optional[str]:
        """Save validation results to CSV files with timestamp.
        
        Returns the path to the validated matches file (medium and high confidence) if available.
        """
        import pandas as pd
        
        validation_path = output_path / "validation"
        validation_path.mkdir(parents=True, exist_ok=True)
        
        validation_df = pd.DataFrame([r.__dict__ for r in validation_results])
        validation_file_path = validation_path / f"validation_results_all_{timestamp}.csv"
        validation_df.to_csv(validation_file_path, index=False)
        logger.info(f"Saved {len(validation_df)} validation results to {validation_file_path}")
        
        # Save medium and high confidence matches and return its path for final report generation
        validated_df = validation_df[validation_df['openai_confidence'].isin(['medium', 'high'])]
        if not validated_df.empty:
            validated_path = validation_path / f"validated_matches_accept_{timestamp}.csv"
            validated_df.to_csv(validated_path, index=False)
            logger.info(f"Saved {len(validated_df)} validated matches (medium/high confidence) to {validated_path}")
            return str(validated_path)
        
        # No medium/high confidence matches - return None (final report won't be generated)
        logger.warning("No medium or high confidence matches found - final report will not be generated")
        return None
    
    def _run_transformation(
        self,
        output_path: Path,
        timestamp: str,
        validation_file: Optional[str],
        filtered_csv: str
    ) -> Optional[str]:
        """Run transformation pipeline."""
        if not self.report_generator:
            return None
        
        final_output_path = output_path / f"final_output_{timestamp}.csv"
        transform_result = self.report_generator.run(
            output_csv_path=str(final_output_path),
            validation_csv_path=validation_file,
            filtered_csv_path=filtered_csv
        )
        
        if transform_result.get('success'):
            logger.info(f"Transformation completed: {final_output_path}")
            return str(final_output_path)
        
        return None

