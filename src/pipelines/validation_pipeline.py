import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

from src.data.models import MatchResult, RestaurantRecord, BusinessRecord
from src.validators.openai_validator import OpenAIValidator, ValidationResult, create_validation_summary

logger = logging.getLogger(__name__)

# Ganesh: separate the OpenAI client and the validation pipeline. The OpenAI client should be a singleton and the validation pipeline should be a class.
class ValidationPipeline:
    """Pipeline for validating matches using OpenAI."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        self.openai_model = openai_model
    
    async def run(
        self,
        input_csv_path: str,
        output_dir: str = "data/output",
        limit: Optional[int] = None,
        batch_size: int = 10,
        max_concurrent_calls: int = 5
    ) -> Dict:
        """
        Run the validation pipeline.
        
        Args:
            input_csv_path: Path to matched restaurants CSV
            output_dir: Directory to save validation results
            limit: Optional limit on number of matches to validate
            batch_size: Batch size for OpenAI API calls
            max_concurrent_calls: Maximum concurrent OpenAI API calls
            
        Returns:
            Dictionary with validation results and output files
        """
        output_path = Path(output_dir) / "validation"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading matches from {input_csv_path}")
        
        try:
            df = pd.read_csv(input_csv_path)
            logger.info(f"Loaded {len(df)} matches")
            if df.empty:
                logger.warning("No matches found in input CSV. Skipping validation.")
                return {
                    'success': False,
                    'error': 'Empty input CSV',
                    'output_files': {}
                }
        except FileNotFoundError:
            logger.error(f"Input CSV not found at {input_csv_path}")
            return {
                'success': False,
                'error': f'File not found: {input_csv_path}',
                'output_files': {}
            }
        except pd.errors.EmptyDataError:
            logger.warning("Input CSV is empty. Skipping validation.")
            return {
                'success': False,
                'error': 'Empty CSV file',
                'output_files': {}
            }
        
        # Reconstruct MatchResult objects from CSV
        matches = self._load_matches_from_csv(df)
        
        if limit:
            matches = matches[:limit]
            logger.info(f"Limited to {len(matches)} matches for validation")
        
        if not matches:
            logger.info("No matches to validate.")
            return {
                'success': False,
                'error': 'No matches to validate',
                'output_files': {}
            }
        
        # Run validation
        validator = OpenAIValidator(
            api_key=self.openai_api_key,
            model=self.openai_model,
            max_concurrent_calls=max_concurrent_calls
        )
        
        logger.info(f"Starting OpenAI validation of {len(matches)} matches")
        validation_results = await validator.validate_matches_batch(matches, batch_size=batch_size)
        
        # Generate output CSVs
        all_results_df = pd.DataFrame([r.__dict__ for r in validation_results])
        all_results_path = output_path / "validated_matches_all.csv"
        all_results_df.to_csv(all_results_path, index=False)
        logger.info(f"Saved all validation results to {all_results_path}")
        
        output_files = {
            'all': str(all_results_path)
        }
        
        accepted_df = all_results_df[all_results_df['openai_recommendation'] == 'accept']
        if not accepted_df.empty:
            accepted_path = output_path / "validated_matches_accept.csv"
            accepted_df.to_csv(accepted_path, index=False)
            logger.info(f"Saved {len(accepted_df)} accept recommendations to {accepted_path}")
            output_files['accept'] = str(accepted_path)
        
        rejected_df = all_results_df[all_results_df['openai_recommendation'] == 'reject']
        if not rejected_df.empty:
            rejected_path = output_path / "validated_matches_reject.csv"
            rejected_df.to_csv(rejected_path, index=False)
            logger.info(f"Saved {len(rejected_df)} reject recommendations to {rejected_path}")
            output_files['reject'] = str(rejected_path)
        
        manual_review_df = all_results_df[all_results_df['openai_recommendation'] == 'manual_review']
        if not manual_review_df.empty:
            manual_review_path = output_path / "validated_matches_manual_review.csv"
            manual_review_df.to_csv(manual_review_path, index=False)
            logger.info(f"Saved {len(manual_review_df)} manual review recommendations to {manual_review_path}")
            output_files['manual_review'] = str(manual_review_path)
        
        # Generate summary report
        summary = create_validation_summary(validation_results)
        summary_report_path = output_path / "validation_summary_report.txt"
        self._write_summary_report(summary_report_path, summary)
        output_files['summary'] = str(summary_report_path)
        
        logger.info("\n✅ Validation completed!")
        logger.info(f"📊 Total validated: {summary['total_validated']}")
        logger.info(f"✅ OpenAI accepts: {summary['accepted_count']} ({summary['accepted_percentage']:.1f}%)")
        logger.info(f"❌ OpenAI rejects: {summary['rejected_count']} ({summary['rejected_percentage']:.1f}%)")
        logger.info(f"🔍 Manual reviews: {summary['manual_review_count']} ({summary['manual_review_percentage']:.1f}%)")
        
        return {
            'success': True,
            'summary': summary,
            'output_files': output_files
        }
    
    def _load_matches_from_csv(self, df: pd.DataFrame) -> List[MatchResult]:
        """Load MatchResult objects from CSV DataFrame."""
        matches = []
        for _, row in df.iterrows():
            restaurant = RestaurantRecord(
                name=row['restaurant_name'],
                address=row['restaurant_address'],
                city=row['restaurant_city'],
                postal_code=str(row['restaurant_postal_code']),
                coordinates=tuple(map(float, row['restaurant_coordinates'].split(', '))),
                rating=row['restaurant_rating'],
                google_id=row.get('restaurant_google_id', ''),
                phone=row.get('restaurant_phone', ''),
                website=row.get('restaurant_website', ''),
                reviews_count=row.get('restaurant_reviews_count'),
                main_type=row.get('restaurant_main_type', '')
            )
            business = BusinessRecord(
                legal_name=row['business_legal_name'],
                registration_number=str(row['business_registration_number']),
                registration_index=row['business_registration_index'],
                business_address=row['business_address'],
                status=row['business_status'],
                resident_agent_name=row.get('business_resident_agent_name', ''),
                resident_agent_address=row.get('business_resident_agent_address', '')
            )
            matches.append(MatchResult(
                restaurant=restaurant,
                business=business,
                confidence_score=row['match_confidence_score'],
                match_type=row['match_type'],
                is_accepted=True,
                name_score=row.get('name_score'),
                postal_code_match=row.get('postal_code_match'),
                city_match=row.get('city_match'),
                match_reason=row.get('match_reason', '')
            ))
        return matches
    
    def _write_summary_report(self, path: Path, summary: Dict) -> None:
        """Write validation summary report to file."""
        with open(path, "w") as f:
            f.write("================================================================================\n")
            f.write("OPENAI VALIDATION REPORT - PUERTO RICO RESTAURANT MATCHER\n")
            f.write("================================================================================\n\n")
            f.write("OVERALL STATISTICS\n")
            f.write("----------------------------------------\n")
            f.write(f"Total matches validated: {summary['total_validated']}\n")
            f.write(f"OpenAI accepts: {summary['accepted_count']} ({summary['accepted_percentage']:.1f}%)\n")
            f.write(f"OpenAI rejects: {summary['rejected_count']} ({summary['rejected_percentage']:.1f}%)\n")
            f.write(f"Manual reviews needed: {summary['manual_review_count']} ({summary['manual_review_percentage']:.1f}%)\n\n")
            
            f.write("CONFIDENCE DISTRIBUTION\n")
            f.write("----------------------------------------\n")
            f.write(f"High confidence: {summary['high_confidence_openai']}\n")
            f.write(f"Medium confidence: {summary['medium_confidence_openai']}\n")
            f.write(f"Low confidence: {summary['low_confidence_openai']}\n\n")
            
            f.write("SCORE COMPARISON\n")
            f.write("----------------------------------------\n")
            f.write(f"Average RapidFuzz score: {summary['avg_rapidfuzz_score']:.1f}%\n")
            f.write(f"Average OpenAI score: {summary['avg_openai_score']:.1f}%\n")
            f.write(f"Score difference: {summary['avg_openai_score'] - summary['avg_rapidfuzz_score']:.1f}%\n\n")
            
            f.write("SAMPLE VALIDATION CASES\n")
            f.write("----------------------------------------\n")
            f.write("High Confidence Accepts:\n")
            for res in summary['sample_accepts']:
                f.write(f"• {res.restaurant_name} → {res.business_legal_name}\n")
                f.write(f"  RapidFuzz: {res.rapidfuzz_confidence_score:.1f}% | OpenAI: {res.openai_match_score}% | {res.openai_reasoning}\n\n")
            
            if summary['sample_rejects']:
                f.write("High Confidence Rejects:\n")
                for res in summary['sample_rejects']:
                    f.write(f"• {res.restaurant_name} → {res.business_legal_name}\n")
                    f.write(f"  RapidFuzz: {res.rapidfuzz_confidence_score:.1f}% | OpenAI: {res.openai_match_score}% | {res.openai_reasoning}\n\n")
            
            if summary['sample_manual_reviews']:
                f.write("Manual Review Cases:\n")
                for res in summary['sample_manual_reviews']:
                    f.write(f"• {res.restaurant_name} → {res.business_legal_name}\n")
                    f.write(f"  RapidFuzz: {res.rapidfuzz_confidence_score:.1f}% | OpenAI: {res.openai_match_score}% | {res.openai_reasoning}\n\n")
            
            f.write("================================================================================\n")
            f.write("END OF VALIDATION REPORT\n")
            f.write("================================================================================\n")
        
        logger.info(f"Validation report generated at {path}")

