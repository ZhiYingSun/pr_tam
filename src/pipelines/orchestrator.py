import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from src.pipelines.matching_pipeline import MatchingPipeline
from src.pipelines.validation_pipeline import ValidationPipeline
from src.pipelines.transformation_pipeline import TransformationPipeline
from src.data.models import MatchingConfig

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
        
        # Initialize pipelines
        self.matching_pipeline = MatchingPipeline(config=config, use_mock=use_mock)
        
        if not skip_validation:
            try:
                self.validation_pipeline = ValidationPipeline()
            except ValueError as e:
                logger.warning(f"Validation pipeline unavailable: {e}")
                self.skip_validation = True
        
        if not skip_transformation:
            self.transformation_pipeline = TransformationPipeline()
    
    def find_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Find the latest file matching a pattern in the directory."""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    
    async def run(
        self,
        input_csv: str,
        output_dir: str = "data/output",
        limit: int = 50,
        batch_size: int = 25,
        max_concurrent: int = 20,
        use_async: bool = True
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Output directory for results
            limit: Number of restaurants to process
            batch_size: Batch size for processing
            max_concurrent: Maximum concurrent API calls
            use_async: Whether to use async processing for matching
            
        Returns:
            Dictionary with complete pipeline results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Input: {input_csv}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Limit: {limit}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  Use async: {use_async}")
        logger.info(f"  Use mock: {self.use_mock}")
        logger.info(f"  Skip validation: {self.skip_validation}")
        logger.info(f"  Skip transformation: {self.skip_transformation}")
        logger.info("=" * 80)
        
        results = {
            'timestamp': timestamp,
            'input_csv': input_csv,
            'output_dir': str(output_path),
            'steps': {}
        }
        
        # Step 1: Matching
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: MATCHING PIPELINE")
        logger.info("=" * 80)
        
        matching_result = await self.matching_pipeline.run(
            input_csv=input_csv,
            output_dir=str(output_path),
            batch_size=batch_size,
            limit=limit,
            use_async=use_async,
            max_concurrent=max_concurrent
        )
        
        results['steps']['matching'] = matching_result
        
        if not matching_result.get('success'):
            logger.error("❌ Matching pipeline failed. Stopping pipeline.")
            return results
        
        # Find the generated matched file
        matched_file = self.find_latest_file(output_path, "matched_restaurants_*.csv")
        if not matched_file:
            logger.error("❌ Could not find generated matched restaurants file")
            results['success'] = False
            return results
        
        logger.info(f"✅ Matching completed: {matched_file}")
        stats = matching_result.get('statistics', {})
        logger.info(f"   Match rate: {stats.get('match_rate', 0):.1f}%")
        logger.info(f"   Matched: {stats.get('matched_count', 0)}/{stats.get('total_count', 0)}")
        results['matched_file'] = str(matched_file)
        
        # Step 2: Validation
        if not self.skip_validation:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: VALIDATION PIPELINE")
            logger.info("=" * 80)
            
            validation_result = await self.validation_pipeline.run(
                input_csv_path=str(matched_file),
                output_dir=str(output_path),
                batch_size=5,
                max_concurrent_calls=3
            )
            
            results['steps']['validation'] = validation_result
            
            if validation_result.get('success'):
                validation_file = output_path / "validation" / "validated_matches_accept.csv"
                if validation_file.exists():
                    logger.info(f"✅ Validation completed: {validation_file}")
                    results['validation_file'] = str(validation_file)
                else:
                    logger.warning("⚠️  No validation accept file generated")
            else:
                logger.warning(f"⚠️  Validation pipeline had issues: {validation_result.get('error', 'Unknown error')}")
        else:
            logger.info("\n⏭️  Skipping validation step")
            results['validation_file'] = None
        
        # Step 3: Transformation
        if not self.skip_transformation:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: TRANSFORMATION PIPELINE")
            logger.info("=" * 80)
            
            validation_file = results.get('validation_file')
            final_output = output_path / f"final_output_{limit}_{timestamp}.csv"
            
            transform_result = self.transformation_pipeline.run(
                input_csv_path=str(matched_file),
                output_csv_path=str(final_output),
                validation_csv_path=validation_file
            )
            
            results['steps']['transformation'] = transform_result
            
            if transform_result.get('success'):
                logger.info(f"✅ Transformation completed: {final_output}")
                logger.info(f"   Records: {transform_result.get('record_count', 0)}")
                results['final_output'] = str(final_output)
            else:
                logger.error(f"❌ Transformation failed: {transform_result.get('error', 'Unknown error')}")
        else:
            logger.info("\n⏭️  Skipping transformation step")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED")
        logger.info("=" * 80)
        
        results['success'] = True
        return results

