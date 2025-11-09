import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from src.utils.loader import load_restaurants
from src.utils.output import generate_all_outputs, get_match_statistics
from src.data.models import MatchingConfig, RestaurantRecord, MatchResult
from src.matchers.async_matcher import AsyncRestaurantMatcher
from src.searchers.async_searcher import AsyncIncorporationSearcher, AsyncMockIncorporationSearcher
from src.validators.llm_validator import LLMValidator, ValidationResult
from src.pipelines.transformation_pipeline import TransformationPipeline

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete pipeline execution."""
    
    def __init__(
        self,
        openai_api_key: str,
        config: Optional[MatchingConfig] = None,
        use_mock: bool = False,
        skip_transformation: bool = False
    ):
        self.config = config or MatchingConfig()
        self.use_mock = use_mock
        self.skip_transformation = skip_transformation

        self.validator = LLMValidator(
            api_key=openai_api_key,
            model="gpt-4o-mini",
            max_concurrent_calls=5
        )
        
        if not skip_transformation:
            self.transformation_pipeline = TransformationPipeline()
        else:
            self.transformation_pipeline = None

    async def run_restaurant(
        self,
        restaurant: RestaurantRecord,
        matcher: AsyncRestaurantMatcher
    ) -> Tuple[Optional[MatchResult], Optional[ValidationResult]]:
        """
        Process a single restaurant through the full match → validate pipeline.

        Runs for one restaurant, but multiple tasks execute concurrently via asyncio.gather().
        All tasks share the same matcher, connection pool (ZyteClient), and rate limiter (matcher.max_concurrent).

        Args:
        restaurant: RestaurantRecord to process.
        matcher: Shared AsyncRestaurantMatcher instance.

        Returns:
        Tuple of (MatchResult, ValidationResult). Either can be None if not found/available.
        """
        # Step 1: Find best match
        match_result = None
        try:
            match_result = await matcher.find_best_match_async(restaurant)
        except Exception as e:
            logger.error(f"Error matching restaurant '{restaurant.name}': {e}",  exc_info=True)

        # Step 2: Validate match (if match found)
        validation_result = None
        if match_result and match_result.business:
            try:
                validation_result = await self.validator.validate_match(match_result)
            except Exception as e:
                logger.error(f"Error validating match for '{restaurant.name}': {e}", exc_info=True)

        return match_result, validation_result
    
    async def run(
        self,
        input_csv: str,
        output_dir: str = "data/output",
        limit: Optional[int] = None,
        max_concurrent: int = 20
    ) -> Dict:
        """
        Run the complete pipeline for all restaurants.
        Uses asyncio.gather at the root to process all restaurants concurrently.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Output directory for results
            limit: Optional limit on number of restaurants to process
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Dictionary with complete pipeline results
        """
        timestamp, output_path, start_time = self._initialize_pipeline_run(
            input_csv, output_dir, limit, max_concurrent
        )
        
        # Load restaurants
        logger.info("Loading restaurants...")
        restaurants = load_restaurants(input_csv, limit=limit)
        logger.info(f"Loaded {len(restaurants)} restaurants")
        
        # Initialize shared searcher and matcher
        if self.use_mock:
            searcher = AsyncMockIncorporationSearcher()
        else:
            zyte_api_key = os.getenv("ZYTE_API_KEY")
            if not zyte_api_key:
                raise ValueError("ZYTE_API_KEY environment variable not set")
            searcher = AsyncIncorporationSearcher(zyte_api_key, max_concurrent=max_concurrent)
        
        async with searcher:
            matcher = AsyncRestaurantMatcher(searcher, max_concurrent=max_concurrent)

            logger.info(f"Processing {len(restaurants)} restaurants concurrently...")
            tasks = [self.run_restaurant(restaurant, matcher) for restaurant in restaurants]
            restaurant_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        match_results = []
        validation_results = []
        error_count = 0
        
        for i, result in enumerate(restaurant_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing restaurant '{restaurants[i].name}': {result}")
                error_count += 1
                continue
            
            match_result, validation_result = result
            
            if match_result:
                match_results.append(match_result)
            
            if validation_result:
                validation_results.append(validation_result)
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
        
        logger.info(f" Processed {len(restaurants)} restaurants")
        logger.info(f"   Matches found: {len(match_results)}")
        logger.info(f"   Validated: {len(validation_results)}")
        
        # Generate output files
        logger.info("Generating output files...")
        output_files = generate_all_outputs(match_results, str(output_path))
        
        # Calculate statistics
        statistics = get_match_statistics(match_results)
        logger.info(f"Match rate: {statistics['match_rate']:.1f}%")
        logger.info(f"Average confidence: {statistics['avg_confidence_score']:.1f}%")
        
        # Save validation results if available
        validation_file = self._save_validation_results(validation_results, output_path) if validation_results else None
        
        # Transformation (if enabled)
        final_output = self._run_transformation(
            output_path, timestamp, validation_file
        ) if not self.skip_transformation and match_results else None
        
        duration = datetime.now() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 80)
        
        return {
            'success': True,
            'timestamp': timestamp,
            'input_csv': input_csv,
            'output_dir': str(output_path),
            'results': match_results,
            'validation_results': validation_results,
            'statistics': statistics,
            'output_files': output_files,
            'matched_file': str(output_path / "matched_restaurants.csv") if match_results else None,
            'validation_file': validation_file,
            'final_output': final_output,
            'duration': duration
        }
    
    def _initialize_pipeline_run(
        self,
        input_csv: str,
        output_dir: str,
        limit: Optional[int],
        max_concurrent: int
    ) -> Tuple[str, Path, datetime]:
        """
        Initialize pipeline run: setup paths, timestamps, and log configuration.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Output directory for results
            limit: Optional limit on number of restaurants
            max_concurrent: Maximum concurrent API calls
            
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
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  Use mock: {self.use_mock}")
        logger.info(f"  Validation: enabled")
        logger.info(f"  Skip transformation: {self.skip_transformation}")
        logger.info("=" * 80)
        
        return timestamp, output_path, start_time
    
    def _save_validation_results(
        self,
        validation_results: List[ValidationResult],
        output_path: Path
    ) -> Optional[str]:
        """Save validation results to CSV files."""
        import pandas as pd
        
        validation_path = output_path / "validation"
        validation_path.mkdir(parents=True, exist_ok=True)
        
        validation_df = pd.DataFrame([r.__dict__ for r in validation_results])
        validation_file_path = validation_path / "validated_matches_all.csv"
        validation_df.to_csv(validation_file_path, index=False)
        
        # Save accepted matches
        accepted_df = validation_df[validation_df['openai_recommendation'] == 'accept']
        if not accepted_df.empty:
            accepted_path = validation_path / "validated_matches_accept.csv"
            accepted_df.to_csv(accepted_path, index=False)
            logger.info(f"Saved {len(accepted_df)} accepted matches to {accepted_path}")
        
        return str(validation_file_path)
    
    def _run_transformation(
        self,
        output_path: Path,
        timestamp: str,
        validation_file: Optional[str]
    ) -> Optional[str]:
        """Run transformation pipeline if enabled."""
        if not self.transformation_pipeline:
            return None
        
        matched_file = output_path / "matched_restaurants.csv"
        if not matched_file.exists():
            return None
        
        final_output_path = output_path / f"final_output_{timestamp}.csv"
        transform_result = self.transformation_pipeline.run(
            input_csv_path=str(matched_file),
            output_csv_path=str(final_output_path),
            validation_csv_path=validation_file
        )
        
        if transform_result.get('success'):
            logger.info(f"✅ Transformation completed: {final_output_path}")
            return str(final_output_path)
        
        return None

