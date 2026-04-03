# Insurance Claims Agent

[![Tests](https://github.com/tyreecepaul/insurance-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/tyreecepaul/insurance-agent/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()

A LangGraph-based conversational agent for automating insurance claim processing.

## Features

- **Policy Search**: Vector + BM25 hybrid search over policy documents
- **Damage Assessment**: CLIP embeddings + BLIP-2 captions for damage photo analysis
- **Claim Processing**: Structured claim drafting with memory persistence
- **Multi-Modal**: Handle text queries, images, and claim status lookups
- **Evaluation Framework**: MLflow + PySpark for variant testing and metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama (for local LLM inference)
- Java (for Spark-based evaluation)

### Installation

```bash
# Clone repo
git clone https://github.com/tyreecepaul/insurance-agent.git
cd insurance-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing
```

### Running the Agent

```bash
# Start Ollama first
ollama serve

# In another terminal, run the agent
python src/agent.py

# Or in Python:
python main.py
```

### Running Tests

```bash
# Unit tests only (fast, <10s)
pytest tests/unit -m unit -v

# Integration tests (comprehensive, ~30s)
pytest tests/integration -m integration -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Documentation

- [Testing Guide](tests/README.md) — Test structure, fixtures, best practices
- [Agent Architecture](docs/ARCHITECTURE.md) — Graph design and node logic (if available)
- [Configuration](config.json) — Model names, API endpoints, paths

## CI/CD

Tests run automatically on:
- Every push to `main` or `develop`
- Every PR to these branches
- View results: [Actions tab](https://github.com/tyreecepaul/insurance-agent/actions)

## Project Structure

```
├── src/
│   ├── agent.py          # LangGraph agent with memory/router/retrieval/generator nodes
│   ├── ingest.py         # ChromaDB indexing for policies/damage/claims
│   ├── eval.py           # MLflow evaluation harness with variant configs
│   ├── tools.py          # Hybrid search, retrieval functions
│   └── test_tools.py     # Utilities for testing and debugging
├── tests/
│   ├── unit/             # Fast unit tests (all mocked)
│   ├── integration/       # E2E tests (mocked external services)
│   ├── fixtures/         # Test data
│   └── conftest.py       # Pytest fixtures and configuration
├── data/
│   ├── policy_docs/      # PDF files to index
│   ├── damage_photos/    # Images for CLIP embedding
│   └── claims_data/      # JSON claims records
├── chroma_db/            # Persistent vector database
├── config.json           # Model configs and API endpoints
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development and test dependencies
└── main.py              # CLI entry point
```

## Development

### Adding Tests

1. Create test in `tests/unit/` or `tests/integration/`
2. Use fixtures from `conftest.py`
3. Mark with `@pytest.mark.unit` or `@pytest.mark.integration`
4. Run: `pytest tests/path/to/test.py -v`

See [Testing Guide](tests/README.md) for examples.

### Code Quality

Format code before commit:
```bash
# Optional linting
flake8 src/ tests/
```

All tests must pass before merging (enforced by CI/CD).

## License

MIT
