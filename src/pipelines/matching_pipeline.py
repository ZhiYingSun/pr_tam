import os
import logging
import asyncio
from typing import List, Optional, Dict
from datetime import datetime

from src.utils.loader import load_restaurants, get_data_summary
from src.utils.output import generate_all_outputs, get_match_statistics
from src.data.models import MatchingConfig, MatchResult, RestaurantRecord
from src.matchers.async_matcher import AsyncRestaurantMatcher
from src.searchers.async_searcher import AsyncIncorporationSearcher, AsyncMockIncorporationSearcher

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """Pipeline for matching restaurants with PR incorporation documents."""
    
    def __init__(
        self,
        config: Optional[MatchingConfig] = None,
        use_mock: bool = False
    ):
        self.config = config or MatchingConfig()
        self.use_mock = use_mock
        
        if use_mock:
            logger.info("Using mock searcher for testing")
        else:
            zyte_api_key = os.getenv('ZYTE_API_KEY')
            if not zyte_api_key:
                raise ValueError("ZYTE_API_KEY environment variable not set")
            logger.info("Using Zyte API searcher")
    
    async def run(
        self,
        input_csv: str,
        output_dir: str = "data/output",
        batch_size: int = 100,
        limit: Optional[int] = None,
        max_concurrent: int = 20
    ) -> Dict:
        """
        Run the matching pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Directory to save output files
            batch_size: Number of restaurants to process per batch
            limit: Optional limit on number of restaurants to process
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Dictionary with results and statistics
        """
        return await self._run_async(
            input_csv=input_csv,
            output_dir=output_dir,
            batch_size=batch_size,
            limit=limit,
            max_concurrent=max_concurrent
        )
    
    async def _run_async(
        self,
        input_csv: str,
        output_dir: str,
        batch_size: int,
        limit: Optional[int],
        max_concurrent: int
    ) -> Dict:
        """Run asynchronous matching pipeline."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting async matching pipeline")
            logger.info(f"Input: {input_csv}")
            logger.info(f"Output: {output_dir}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Max concurrent: {max_concurrent}")
            if limit:
                logger.info(f"Limit: {limit} restaurants")
            
            # Step 1: Load restaurants
            logger.info("Loading restaurants...")
            restaurants = load_restaurants(input_csv, limit=limit)
            logger.info(f"Loaded {len(restaurants)} restaurants")
            
            # Step 2: Process restaurants with async concurrent matching
            logger.info("Processing restaurants with async concurrent matching...")
            all_results = []
            
            for i in range(0, len(restaurants), batch_size):
                batch_restaurants = restaurants[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(restaurants) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}: restaurants {i+1}-{min(i + batch_size, len(restaurants))}")
                
                if self.use_mock:
                    async with AsyncMockIncorporationSearcher() as searcher:
                        matcher = AsyncRestaurantMatcher(searcher, max_concurrent=max_concurrent)
                        batch_results = await matcher.match_multiple_restaurants_async(batch_restaurants)
                else:
                    zyte_api_key = os.getenv("ZYTE_API_KEY")
                    if not zyte_api_key:
                        raise ValueError("ZYTE_API_KEY environment variable not set")
                    
                    async with AsyncIncorporationSearcher(zyte_api_key, max_concurrent=max_concurrent) as searcher:
                        matcher = AsyncRestaurantMatcher(searcher, max_concurrent=max_concurrent)
                        batch_results = await matcher.match_multiple_restaurants_async(batch_restaurants)
                
                all_results.extend(batch_results)
                
                progress = (i + len(batch_restaurants)) / len(restaurants) * 100
                logger.info(f"Progress: {i + len(batch_restaurants)}/{len(restaurants)} ({progress:.1f}%)")
                logger.info(f"Batch completed: {len(batch_results)}/{len(batch_restaurants)} matched")
            
            logger.info(f"Completed processing {len(restaurants)} restaurants")
            
            # Step 3: Generate output files
            logger.info("Generating output files...")
            output_files = generate_all_outputs(all_results, output_dir)
            
            # Step 4: Calculate statistics
            logger.info("Calculating statistics...")
            statistics = get_match_statistics(all_results)
            
            duration = datetime.now() - start_time
            logger.info(f"Pipeline completed in {duration}")
            logger.info(f"Match rate: {statistics['match_rate']:.1f}%")
            logger.info(f"Average confidence: {statistics['avg_confidence_score']:.1f}%")
            
            return {
                'success': True,
                'results': all_results,
                'statistics': statistics,
                'output_files': output_files,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'duration': datetime.now() - start_time
            }

