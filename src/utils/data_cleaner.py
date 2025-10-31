import pandas as pd
from pathlib import Path


def clean_restaurant_data(input_file: str, output_file: str) -> None:
    """
    Filter out closed restaurants from Google Maps data.
    
    Args:
        input_file: Path to the raw CSV file
        output_file: Path to save the cleaned CSV file
    """
    # Read data
    df = pd.read_csv(input_file)
    
    # Filter to keep only open restaurants
    cleaned_df = df[df['Is closed'] == 'No'].copy()
    
    # Save cleaned data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    
    print(f"Filtered {len(df)} -> {len(cleaned_df)} restaurants")


if __name__ == "__main__":
    input_file = "data/raw/Puerto Rico Data.csv"
    output_file = "data/processed/cleaned_restaurants.csv"
    
    clean_restaurant_data(input_file, output_file)
