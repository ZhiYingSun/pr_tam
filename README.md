# TAM PR

A high-performance Python application that matches restaurant data with Puerto Rico incorporation documents.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete pipeline (filtering is enabled by default)
python main.py --limit 50

# Process entire dataset with filtering
python main.py --input data/Puerto\ Rico\ Data_\ v1109_155.csv

# Skip filtering if working with pre-filtered data
python main.py --limit 50 --skip-filter

# Use custom filter lists
python main.py \
  --exclusion-list src/misc/excluded_business_types.txt \
  --inclusion-list src/misc/included_business_types.txt
```

## Project Structure

```
pr_tam/
├── main.py          # Main entry point for the pipeline
├── src/             # Core source code
│   ├── orchestrator/   # Pipeline orchestration
│   ├── matchers/    # Matching algorithms
│   ├── searchers/   # Business search logic
│   ├── validators/  # OpenAI validation
│   ├── utils/       # Utility functions
│   └── models/      # Data models
├── data/            # Data directories (raw, processed, output)
├── tests/           # Test suite
├── config/          # Configuration files
├── logs/            # Log files
└── docs/            # Documentation
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file with your API keys)
# Or export them directly:
export ZYTE_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

## Business Type Filtering

The pipeline **automatically applies business type filtering by default** to remove unsupported business types (bars, clubs, gyms, etc.) and keep only food-related businesses. This ensures DoorDash-compatible data processing.

**How it works:**
- Filters businesses based on exclusion and inclusion lists
- Logic: Remove if business matches exclusion type AND does NOT match any inclusion type
- Keeps mixed-type businesses (e.g., "Bar + Restaurant") if they have at least one inclusion type

**Usage:**
```bash
# Filtering is enabled by default - just run normally
python main.py --limit 50

# Skip filtering if working with pre-filtered data
python main.py --limit 50 --skip-filter

# Use custom filter lists
python main.py \
  --exclusion-list path/to/excluded.txt \
  --inclusion-list path/to/included.txt
```

**Filter Lists:**
- Exclusion list: `src/misc/excluded_business_types.txt` (221 types: bars, clubs, gyms, services, etc.)
- Inclusion list: `src/misc/included_business_types.txt` (257 types: restaurants, cafes, food stores, etc.)

**Output:**
- Filtered businesses saved to `data/output/filtered/filtered_businesses_{timestamp}.csv`
- Removed businesses saved to `data/output/filtered/removed_businesses_{timestamp}.csv`

## Command Line Arguments

```
--input, -i           Input CSV file path (default: data/Puerto Rico Data_ v1109_155.csv)
--output, -o          Output directory (default: data/output)
--limit, -l           Number of restaurants to process (default: 50)
--threshold, -t       Name match threshold percentage (default: 70.0)
--skip-filter         Skip business type filtering (enabled by default)
--exclusion-list      Path to exclusion list for filtering (default: src/misc/excluded_business_types.txt)
--inclusion-list      Path to inclusion list for filtering (default: src/misc/included_business_types.txt)
--verbose, -v         Enable verbose logging
```

For complete documentation, see [docs/README.md](docs/README.md).

