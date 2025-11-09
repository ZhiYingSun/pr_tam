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
from src.searchers.searcher import IncorporationSearcher
from src.clients.openai_client import OpenAIClient

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
  # Production run with 50 restaurants
  python main.py --limit 50 --max-concurrent 20
  
  # Full pipeline with all steps
  python main.py --limit 500 --max-concurrent 20
  
  # Skip transformation for faster testing
  python main.py --limit 100 --skip-transformation
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
        '--skip-transformation',
        action='store_true',
        help='Skip data transformation step'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = MatchingConfig()
    config.NAME_MATCH_THRESHOLD = args.threshold
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Validate API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment. Validation is required.")
        sys.exit(1)
    
    zyte_api_key = os.getenv("ZYTE_API_KEY")
    if not zyte_api_key:
        logger.error("ZYTE_API_KEY not found in environment.")
        sys.exit(1)
    
    # Create singleton clients (rate limited)
    openai_client = OpenAIClient(api_key=openai_api_key)
    searcher = IncorporationSearcher(zyte_api_key, max_concurrent=args.max_concurrent)
 
    try:
        orchestrator = PipelineOrchestrator(
            openai_client=openai_client,
            searcher=searcher,
            config=config,
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
            max_concurrent=args.max_concurrent
        ))
        
        # Check success status (orchestrator raises exception if error_rate > 50%)
        if result.get('success'):
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✅ Pipeline completed successfully")
            logger.info(f"📁 Output directory: {result['output_dir']}")
            
            error_count = result.get('error_count', 0)
            error_rate = result.get('error_rate', 0.0)
            if error_count > 0:
                logger.info(f"⚠️  Errors encountered: {error_count} ({error_rate*100:.1f}% error rate)")
            
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

