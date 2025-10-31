#!/usr/bin/env python3
"""
Puerto Rico Restaurant Matcher - Main Entry Point
Complete end-to-end pipeline for matching restaurants with PR incorporation documents
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.pipelines.orchestrator import PipelineOrchestrator
from src.data.models import MatchingConfig

# Configure logging
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Puerto Rico Restaurant Matcher - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 restaurants (mock data)
  python main.py --limit 5 --mock
  
  # Production run with 50 restaurants
  python main.py --limit 50 --batch-size 25 --max-concurrent 20
  
  # Full pipeline with all steps
  python main.py --limit 500 --batch-size 100 --max-concurrent 20
  
  # Skip validation for faster testing
  python main.py --limit 100 --skip-validation
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='data/processed/cleaned_restaurants.csv',
        help='Input CSV file path (default: data/processed/cleaned_restaurants.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/output',
        help='Output directory (default: data/output)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=50,
        help='Number of restaurants to process (default: 50)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=25,
        help='Batch size for processing (default: 25)'
    )
    parser.add_argument(
        '--max-concurrent', '-c',
        type=int,
        default=20,
        help='Maximum concurrent API calls (default: 20)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=70.0,
        help='Name match threshold percentage (default: 70.0)'
    )
    parser.add_argument(
        '--mock', '-m',
        action='store_true',
        help='Use mock searcher for testing (no API calls)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip OpenAI validation step'
    )
    parser.add_argument(
        '--skip-transformation',
        action='store_true',
        help='Skip data transformation step'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Use synchronous processing instead of async'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = MatchingConfig()
    config.NAME_MATCH_THRESHOLD = args.threshold
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(
            config=config,
            use_mock=args.mock,
            skip_validation=args.skip_validation,
            skip_transformation=args.skip_transformation
        )
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        result = asyncio.run(orchestrator.run(
            input_csv=str(input_path),
            output_dir=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            use_async=not args.sync
        ))
        
        if result.get('success'):
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✅ Pipeline completed successfully")
            logger.info(f"📁 Output directory: {result['output_dir']}")
            
            if result.get('matched_file'):
                logger.info(f"📄 Matched restaurants: {result['matched_file']}")
            
            if result.get('validation_file'):
                logger.info(f"📄 Validation results: {result['validation_file']}")
            
            if result.get('final_output'):
                logger.info(f"📄 Final output: {result['final_output']}")
            
            logger.info("=" * 80)
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

