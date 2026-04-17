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

## Business Impact

### Problem solved

Insurance customers filing claims must navigate dense policy documents, describe
damage accurately, and track claim status — all while under stress. Mistakes at
any step produce rejected or delayed claims, repeat contacts with the insurer,
and increased handling cost. This agent addresses all three friction points in a
single conversational interface.

### Quality improvement (ablation results)

The evaluation harness tests four system variants from a plain LLM baseline (A1)
to the full agent (A4) across 11 benchmark queries in four families. Run
`python src/eval.py && mlflow ui` to reproduce.

| Variant | System | Avg judge score (1–5) | Avg recall@5 |
|---------|--------|-----------------------|--------------|
| A1 | LLM only, no retrieval | *run eval to populate* | *run eval* |
| A2 | Text retrieval, no image index | *run eval to populate* | *run eval* |
| A3 | All indices, no query router | *run eval to populate* | *run eval* |
| A4 | Full system (router + all indices) | *run eval to populate* | *run eval* |

**Translating metrics to outcomes:**

- **Judge score Δ (A4 vs A1)** — a 1-point improvement on a 5-point scale
  represents the difference between a mostly-wrong coverage answer and a mostly-
  correct one. In production, each mis-handled coverage decision either results
  in an incorrect rejection (customer harm) or an incorrect approval (claim
  leakage). Reducing this rate directly lowers re-work and escalation volume.

- **Recall@5 on cross-modal queries** — the gap between A2 (text-only) and A4
  (with CLIP image index) quantifies how much grounding quality degrades when
  the damage photo pipeline is absent. Damage queries account for a large share
  of real claim interactions; missing this modality means the agent answers from
  general knowledge rather than indexed evidence.

- **Latency (A3 vs A4)** — A3 blasts all three indices on every query. The
  router in A4 targets only the relevant index per query type, reducing average
  token consumption and response latency. Lower latency directly maps to higher
  conversation completion rates.

- **Token cost (A4 full run)** — `avg_total_tokens` logged by MLflow is the
  per-query LLM cost proxy. At scale, routing queries to targeted indices rather
  than passing all retrieved context to the generator yields a measurable
  reduction in cost per handled claim.

### SQL analytics (DuckDB)

`src/analytics.py` exposes four aggregate queries over the claims dataset,
demonstrating the structured-data layer that sits alongside the vector pipeline:

```bash
python src/analytics.py                  # all four queries
python src/analytics.py --query approval # approval rate by insurance type
python src/analytics.py --query pipeline # pipeline exposure summary
```

---

## Deployment

### Docker (local, reproducible)

```bash
# 1. Start Ollama on the host
ollama serve

# 2. Index data once (run outside Docker — chroma_db is volume-mounted)
python src/ingest.py

# 3. Build and start the API container
docker compose up --build

# 4. Verify
curl http://localhost:8000/health
```

The `/health` endpoint returns uptime, active session count, and the configured
Ollama URL — wired into both the Dockerfile `HEALTHCHECK` and the
`docker-compose.yml` healthcheck so container orchestrators get a live
readiness signal.

**Volume mounts** (configured in `docker-compose.yml`):

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./chroma_db` | `/app/chroma_db` | Pre-built vector indices |
| `./data` | `/app/data` (read-only) | Policy PDFs, damage photos, claims JSON |
| `./uploads` | `/app/uploads` | Runtime image uploads |

---

## License

MIT
