import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from src.utils.loader import load_restaurants
from src.utils.output import generate_all_outputs, get_match_statistics
from src.data.models import MatchingConfig, RestaurantRecord, MatchResult
from src.matchers.async_matcher import AsyncRestaurantMatcher
from src.validators.openai_validator import OpenAIValidator, ValidationResult
from src.pipelines.transformation_pipeline import TransformationPipeline

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete pipeline execution."""
    
    def __init__(
        self,
        searcher,
        config: Optional[MatchingConfig] = None,
        validator: Optional[OpenAIValidator] = None,
        skip_validation: bool = False,
        skip_transformation: bool = False
    ):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            searcher: AsyncIncorporationSearcher instance (or mock for testing).
                     Must implement async context manager and search_business_async().
            config: Optional matching configuration
            validator: Optional OpenAIValidator instance (or mock for testing).
                      If None and skip_validation=False, will attempt to create from env.
            skip_validation: Whether to skip OpenAI validation step
            skip_transformation: Whether to skip data transformation step
        """
        self.searcher = searcher
        self.config = config or MatchingConfig()
        self.skip_validation = skip_validation
        self.skip_transformation = skip_transformation
        
        if validator is not None:
            # Use injected validator
            self.validator = validator
            self.skip_validation = False  # Override skip_validation if validator provided
        elif not skip_validation:
            # Try to create validator from environment if not provided
            try:
                openai_api_key = os.getenv('OPENAI_API_KEY')
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.validator = OpenAIValidator(
                    api_key=openai_api_key,
                    model="gpt-4o-mini",
                    max_concurrent_calls=5
                )
            except ValueError as e:
                logger.warning(f"Validation unavailable: {e}")
                self.skip_validation = True
                self.validator = None
        else:
            self.validator = None
        
        if not skip_transformation:
            self.transformation_pipeline = TransformationPipeline()
        else:
            self.transformation_pipeline = None
    
    def find_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Find the latest file matching a pattern in the directory."""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    
    async def run_restaurant(
        self,
        restaurant: RestaurantRecord,
        matcher: AsyncRestaurantMatcher
    ) -> Dict:
        """
        Process a single restaurant end-to-end: match -> validate -> return result.
        
        This method processes ONE restaurant through the complete pipeline.
        All restaurants are processed concurrently via asyncio.gather() at the root level.
        
        The matcher (and its underlying searcher) are shared across all restaurants:
        - Single connection pool (via ZyteClient singleton)
        - Single semaphore for rate limiting (via matcher.max_concurrent)
        - All async operations happen concurrently at the root, not in nested batches
        
        Args:
            restaurant: RestaurantRecord to process
            matcher: AsyncRestaurantMatcher instance (shared across all restaurants)
            
        Returns:
            Dictionary with match_result and validation_result (if validation enabled)
        """
        # Step 1: Find best match
        match_result = None
        try:
            match_result = await matcher.find_best_match_async(restaurant)
        except Exception as e:
            logger.error(f"Error matching restaurant '{restaurant.name}': {e}")
        
        # Step 2: Validate match (if enabled and match found)
        validation_result = None
        if not self.skip_validation and self.validator and match_result and match_result.business:
            try:
                validation_result = await self.validator.validate_match(match_result)
            except Exception as e:
                logger.error(f"Error validating match for '{restaurant.name}': {e}")
        
        return {
            'match_result': match_result,
            'validation_result': validation_result
        }
    
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Input: {input_csv}")
        logger.info(f"  Output: {output_dir}")
        if limit:
            logger.info(f"  Limit: {limit} restaurants")
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  Skip validation: {self.skip_validation}")
        logger.info(f"  Skip transformation: {self.skip_transformation}")
        logger.info("=" * 80)
        
        # Load restaurants
        logger.info("Loading restaurants...")
        restaurants = load_restaurants(input_csv, limit=limit)
        logger.info(f"Loaded {len(restaurants)} restaurants")
        
        # Use the injected searcher (shared across ALL restaurants for connection pooling and rate limiting)
        async with self.searcher:
            # Create a single matcher instance shared across all restaurants
            # The matcher's semaphore limits concurrent API calls across all restaurants
            matcher = AsyncRestaurantMatcher(self.searcher, max_concurrent=max_concurrent)
            
            # Process all restaurants concurrently using asyncio.gather at the ROOT level
            # This is the ONLY place where async concurrency happens - no nested batching
            # Each restaurant goes through: match -> validate (sequential for that restaurant)
            # But all restaurants run concurrently via gather()
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
            
            if result.get('match_result'):
                match_results.append(result['match_result'])
            
            if result.get('validation_result'):
                validation_results.append(result['validation_result'])
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
        
        logger.info(f"✅ Processed {len(restaurants)} restaurants")
        logger.info(f"   Matches found: {len(match_results)}")
        if not self.skip_validation:
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
        logger.info("✅ PIPELINE COMPLETED")
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

