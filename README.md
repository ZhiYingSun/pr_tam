# TAM PR

A high-performance Python application that matches restaurant data with Puerto Rico incorporation documents.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete pipeline
python main.py --limit 50 --batch-size 25 --max-concurrent 20

# Quick test with mock data
python main.py --limit 5 --mock --skip-validation
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

For complete documentation, see [docs/README.md](docs/README.md).

