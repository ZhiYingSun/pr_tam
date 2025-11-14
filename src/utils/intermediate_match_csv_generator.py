"""
Intermediate match CSV generation utilities for Puerto Rico Restaurant Matcher.

Generates intermediate CSV files and reports from match results before final transformation.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from src.models.models import MatchResult, RestaurantRecord, BusinessRecord, GeneratedOutputFiles

logger = logging.getLogger(__name__)


def generate_matched_restaurants_csv(matches: List[MatchResult], output_path: str) -> None:
    """
    Generate CSV file with matched restaurants and their business information.
    
    Args:
        matches: List of MatchResult objects (only accepted matches)
        output_path: Path to save the CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter only accepted matches
    accepted_matches = [match for match in matches if match.is_accepted and match.business]
    
    logger.info(f"Generating matched restaurants CSV with {len(accepted_matches)} matches")
    
    # Prepare data for CSV
    data = []
    for match in accepted_matches:
        restaurant = match.restaurant
        business = match.business
        
        row = {
            # Restaurant information (from Google Maps)
            'restaurant_name': restaurant.name,
            'restaurant_address': restaurant.address,
            'restaurant_city': restaurant.city,
            'restaurant_postal_code': restaurant.postal_code,
            'restaurant_coordinates': f"{restaurant.coordinates[0]}, {restaurant.coordinates[1]}",
            'restaurant_rating': restaurant.rating,
            'restaurant_reviews_count': restaurant.reviews_count,
            'restaurant_phone': restaurant.phone,
            'restaurant_website': restaurant.website,
            'restaurant_main_type': restaurant.main_type,
            'restaurant_google_id': restaurant.google_id,
            
            # Business information (from PR incorporation docs)
            'business_legal_name': business.legal_name,
            'business_registration_number': business.registration_number,
            'business_registration_index': business.registration_index,
            'business_address': business.business_address,
            'business_status': business.status,
            'business_resident_agent_name': business.resident_agent_name,
            'business_resident_agent_address': business.resident_agent_address,
            
            # Match information
            'match_confidence_score': match.confidence_score,
            'match_type': match.match_type,
            'name_score': match.name_score,
            'postal_code_match': match.postal_code_match,
            'city_match': match.city_match,
            'match_reason': match.match_reason
        }
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(accepted_matches)} matched restaurants to {output_path}")


def generate_unmatched_restaurants_csv(matches: List[MatchResult], output_path: str) -> None:
    """
    Generate CSV file with unmatched restaurants.
    
    Args:
        matches: List of MatchResult objects (including unmatched)
        output_path: Path to save the CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter unmatched restaurants
    unmatched = [match for match in matches if not match.is_accepted]
    
    logger.info(f"Generating unmatched restaurants CSV with {len(unmatched)} restaurants")
    
    # Prepare data for CSV
    data = []
    for match in unmatched:
        restaurant = match.restaurant
        
        row = {
            'restaurant_name': restaurant.name,
            'restaurant_address': restaurant.address,
            'restaurant_city': restaurant.city,
            'restaurant_postal_code': restaurant.postal_code,
            'restaurant_coordinates': f"{restaurant.coordinates[0]}, {restaurant.coordinates[1]}",
            'restaurant_rating': restaurant.rating,
            'restaurant_reviews_count': restaurant.reviews_count,
            'restaurant_phone': restaurant.phone,
            'restaurant_website': restaurant.website,
            'restaurant_main_type': restaurant.main_type,
            'restaurant_google_id': restaurant.google_id,
            'match_confidence_score': match.confidence_score if match.confidence_score else 0,
            'match_type': match.match_type,
            'match_reason': match.match_reason,
            # Include best candidate business info if available
            'best_candidate_legal_name': match.business.legal_name if match.business else '',
            'best_candidate_registration_number': match.business.registration_number if match.business else '',
            'best_candidate_registration_index': match.business.registration_index if match.business else '',
            'best_candidate_address': match.business.business_address if match.business else '',
            'best_candidate_status': match.business.status if match.business else '',
            'best_candidate_resident_agent_name': match.business.resident_agent_name if match.business else '',
            'best_candidate_resident_agent_address': match.business.resident_agent_address if match.business else '',
            'name_score': match.name_score if match.name_score else 0,
            'postal_code_match': match.postal_code_match if match.postal_code_match is not None else False,
            'city_match': match.city_match if match.city_match is not None else False
        }
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(unmatched)} unmatched restaurants to {output_path}")


def generate_summary_report(matches: List[MatchResult], output_path: str) -> None:
    """
    Generate summary report with statistics and analysis.
    
    Args:
        matches: List of MatchResult objects
        output_path: Path to save the report file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    total_restaurants = len(matches)
    accepted_matches = [match for match in matches if match.is_accepted and match.business]
    unmatched = [match for match in matches if not match.is_accepted]
    
    match_rate = (len(accepted_matches) / total_restaurants * 100) if total_restaurants > 0 else 0
    
    # Confidence score distribution
    confidence_scores = [match.confidence_score for match in accepted_matches if match.confidence_score]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    high_confidence = len([m for m in accepted_matches if m.match_type == "high"])
    medium_confidence = len([m for m in accepted_matches if m.match_type == "medium"])
    low_confidence = len([m for m in accepted_matches if m.match_type == "low"])
    
    # Location match statistics
    postal_code_matches = len([m for m in accepted_matches if m.postal_code_match]) if accepted_matches else 0
    city_matches = len([m for m in accepted_matches if m.city_match]) if accepted_matches else 0
    
    # Generate report
    report_lines = [
        "=" * 80,
        "PUERTO RICO RESTAURANT MATCHER - SUMMARY REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL STATISTICS",
        "-" * 40,
        f"Total restaurants processed: {total_restaurants:,}",
        f"Successfully matched: {len(accepted_matches):,}",
        f"Unmatched: {len(unmatched):,}",
        f"Match rate: {match_rate:.1f}%",
        "",
        "CONFIDENCE DISTRIBUTION",
        "-" * 40,
        f"High confidence matches (≥85%): {high_confidence:,}",
        f"Medium confidence matches (75-84%): {medium_confidence:,}",
        f"Low confidence matches (70-74%): {low_confidence:,}",
        f"Average confidence score: {avg_confidence:.1f}%",
        "",
        "LOCATION MATCHING",
        "-" * 40,
        f"Postal code matches: {postal_code_matches:,} ({postal_code_matches/len(accepted_matches)*100:.1f}% of matches)" if accepted_matches else "Postal code matches: 0 (0.0% of matches)",
        f"City matches: {city_matches:,} ({city_matches/len(accepted_matches)*100:.1f}% of matches)" if accepted_matches else "City matches: 0 (0.0% of matches)",
        "",
        "SAMPLE HIGH-CONFIDENCE MATCHES",
        "-" * 40,
    ]
    
    # Add sample matches
    high_conf_matches = [m for m in accepted_matches if m.match_type == "high"][:5]
    for i, match in enumerate(high_conf_matches, 1):
        report_lines.extend([
            f"{i}. {match.restaurant.name}",
            f"   → {match.business.legal_name}",
            f"   Confidence: {match.confidence_score:.1f}%",
            f"   Location: {match.restaurant.city}, {match.restaurant.postal_code}",
            ""
        ])
    
    report_lines.extend([
        "SAMPLE UNMATCHED RESTAURANTS",
        "-" * 40,
    ])
    
    # Add sample unmatched
    sample_unmatched = unmatched[:5]
    for i, match in enumerate(sample_unmatched, 1):
        report_lines.extend([
            f"{i}. {match.restaurant.name}",
            f"   Location: {match.restaurant.city}, {match.restaurant.postal_code}",
            f"   Reason: {match.match_reason}",
            ""
        ])
    
    report_lines.extend([
        "METHODOLOGY",
        "-" * 40,
        "• Name matching using RapidFuzz token_sort_ratio",
        "• Postal code exact match bonus: +20 points",
        "• City exact match bonus: +10 points",
        "• Minimum confidence threshold: 70%",
        "• High confidence threshold: 85%",
        "• Medium confidence threshold: 75%",
        "",
        "DATA SOURCES",
        "-" * 40,
        "• Restaurant data: Google Maps (Puerto Rico)",
        "• Business data: Puerto Rico incorporation documents",
        "• API: Puerto Rico State Department of State",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Generated summary report at {output_path}")


def generate_all_outputs(matches: List[MatchResult], output_dir: str = "data/output") -> GeneratedOutputFiles:
    """
    Generate all output files (matched CSV, unmatched CSV).
    
    Args:
        matches: List of MatchResult objects
        output_dir: Directory to save output files
        
    Returns:
        GeneratedOutputFiles with file paths of generated files
    """
    logger.info("Generating output files...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File paths
    matched_csv = output_path / f"matched_restaurants_{timestamp}.csv"
    unmatched_csv = output_path / f"unmatched_restaurants_{timestamp}.csv"
    
    # Generate files
    generate_matched_restaurants_csv(matches, str(matched_csv))
    generate_unmatched_restaurants_csv(matches, str(unmatched_csv))
    
    generated_files = GeneratedOutputFiles(
        matched_csv=str(matched_csv),
        unmatched_csv=str(unmatched_csv),
        summary_report=""  # Summary report generation removed
    )
    
    logger.info(f"Generated all output files in {output_dir}")
    return generated_files


def print_match_statistics(matches: List[MatchResult]) -> Dict[str, Any]:
    """
    Get detailed statistics about the matching results.
    
    Args:
        matches: List of MatchResult objects
        
    Returns:
        Dictionary with detailed statistics
    """
    total_restaurants = len(matches)
    accepted_matches = [match for match in matches if match.is_accepted and match.business]
    unmatched = [match for match in matches if not match.is_accepted]
    
    # Basic stats
    stats = {
        'total_restaurants': total_restaurants,
        'matched_restaurants': len(accepted_matches),
        'unmatched_restaurants': len(unmatched),
        'match_rate': (len(accepted_matches) / total_restaurants * 100) if total_restaurants > 0 else 0
    }
    
    if accepted_matches:
        # Confidence stats
        confidence_scores = [match.confidence_score for match in accepted_matches if match.confidence_score]
        stats.update({
            'avg_confidence_score': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'min_confidence_score': min(confidence_scores) if confidence_scores else 0,
            'max_confidence_score': max(confidence_scores) if confidence_scores else 0,
            'high_confidence_count': len([m for m in accepted_matches if m.match_type == "high"]),
            'medium_confidence_count': len([m for m in accepted_matches if m.match_type == "medium"]),
            'low_confidence_count': len([m for m in accepted_matches if m.match_type == "low"])
        })
        
        # Location stats
        stats.update({
            'postal_code_matches': len([m for m in accepted_matches if m.postal_code_match]),
            'city_matches': len([m for m in accepted_matches if m.city_match])
        })
    else:
        # No matches found
        stats.update({
            'avg_confidence_score': 0,
            'min_confidence_score': 0,
            'max_confidence_score': 0,
            'high_confidence_count': 0,
            'medium_confidence_count': 0,
            'low_confidence_count': 0,
            'postal_code_matches': 0,
            'city_matches': 0
        })

    logger.info(f"Match rate: {stats['match_rate']:.1f}%")
    logger.info(f"Average confidence: {stats['avg_confidence_score']:.1f}%")
    
    return stats
