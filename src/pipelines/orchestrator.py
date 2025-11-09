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
from src.searchers.async_searcher import AsyncIncorporationSearcher, AsyncMockIncorporationSearcher
from src.validators.openai_validator import OpenAIValidator, ValidationResult
from src.pipelines.transformation_pipeline import TransformationPipeline

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete pipeline execution."""
    
    def __init__(
        self,
        config: Optional[MatchingConfig] = None,
        use_mock: bool = False,
        skip_validation: bool = False,
        skip_transformation: bool = False
    ):
        self.config = config or MatchingConfig()
        self.use_mock = use_mock
        self.skip_validation = skip_validation
        self.skip_transformation = skip_transformation
        
        if not skip_validation:
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
        searcher,
        matcher: AsyncRestaurantMatcher
    ) -> Dict:
        """
        Process a single restaurant end-to-end: match -> validate -> return result.
        
        Args:
            restaurant: RestaurantRecord to process
            searcher: AsyncIncorporationSearcher instance (shared across all restaurants)
            matcher: AsyncRestaurantMatcher instance (shared across all restaurants)
            
        Returns:
            Dictionary with match_result and validation_result (if validation enabled)
        """
        result = {
            'restaurant': restaurant,
            'match_result': None,
            'validation_result': None
        }
        
        # Step 1: Match
        try:
            match_result = await matcher.find_best_match_async(restaurant)
            result['match_result'] = match_result
        except Exception as e:
            logger.error(f"Error matching restaurant '{restaurant.name}': {e}")
            result['match_result'] = None
        
        # Step 2: Validate (if enabled and match found)
        if not self.skip_validation and self.validator and result['match_result'] and result['match_result'].business:
            try:
                validation_result = await self.validator.validate_match(result['match_result'])
                result['validation_result'] = validation_result
            except Exception as e:
                logger.error(f"Error validating match for '{restaurant.name}': {e}")
                result['validation_result'] = None
        
        return result
    
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
        logger.info(f"  Use mock: {self.use_mock}")
        logger.info(f"  Skip validation: {self.skip_validation}")
        logger.info(f"  Skip transformation: {self.skip_transformation}")
        logger.info("=" * 80)
        
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
            
            # Process all restaurants concurrently using asyncio.gather at root
            logger.info(f"Processing {len(restaurants)} restaurants concurrently...")
            tasks = [self.run_restaurant(restaurant, searcher, matcher) for restaurant in restaurants]
            restaurant_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        all_results = []
        validation_results = []
        error_count = 0
        
        for i, result in enumerate(restaurant_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing restaurant '{restaurants[i].name}': {result}")
                error_count += 1
                continue
            
            if result.get('match_result'):
                all_results.append(result['match_result'])
            
            if result.get('validation_result'):
                validation_results.append(result['validation_result'])
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
        
        logger.info(f"✅ Processed {len(restaurants)} restaurants")
        logger.info(f"   Matches found: {len(all_results)}")
        if not self.skip_validation:
            logger.info(f"   Validated: {len(validation_results)}")
        
        # Generate output files
        logger.info("Generating output files...")
        output_files = generate_all_outputs(all_results, str(output_path))
        
        # Calculate statistics
        statistics = get_match_statistics(all_results)
        logger.info(f"Match rate: {statistics['match_rate']:.1f}%")
        logger.info(f"Average confidence: {statistics['avg_confidence_score']:.1f}%")
        
        # Save validation results if available
        validation_file = None
        if validation_results and not self.skip_validation:
            import pandas as pd
            validation_path = output_path / "validation"
            validation_path.mkdir(parents=True, exist_ok=True)
            
            validation_df = pd.DataFrame([r.__dict__ for r in validation_results])
            validation_file_path = validation_path / "validated_matches_all.csv"
            validation_df.to_csv(validation_file_path, index=False)
            validation_file = str(validation_file_path)
            
            # Save accepted matches
            accepted_df = validation_df[validation_df['openai_recommendation'] == 'accept']
            if not accepted_df.empty:
                accepted_path = validation_path / "validated_matches_accept.csv"
                accepted_df.to_csv(accepted_path, index=False)
                logger.info(f"Saved {len(accepted_df)} accepted matches to {accepted_path}")
        
        # Transformation (if enabled)
        final_output = None
        if not self.skip_transformation and self.transformation_pipeline and all_results:
            matched_file = output_path / "matched_restaurants.csv"
            if matched_file.exists():
                final_output_path = output_path / f"final_output_{timestamp}.csv"
                transform_result = self.transformation_pipeline.run(
                    input_csv_path=str(matched_file),
                    output_csv_path=str(final_output_path),
                    validation_csv_path=validation_file
                )
                if transform_result.get('success'):
                    final_output = str(final_output_path)
                    logger.info(f"✅ Transformation completed: {final_output}")
        
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
            'results': all_results,
            'validation_results': validation_results,
            'statistics': statistics,
            'output_files': output_files,
            'matched_file': str(output_path / "matched_restaurants.csv") if all_results else None,
            'validation_file': validation_file,
            'final_output': final_output,
            'duration': duration
        }

