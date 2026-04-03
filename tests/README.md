# Test Suite for Insurance Agent

This directory contains all tests for the insurance claims agent application.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures (ChromaDB, Ollama, models)
├── unit/                    # Fast unit tests (all deps mocked)
│   ├── test_ingest_utils.py      # chunk_text, classifiers
│   ├── test_agent_nodes.py       # memory_node, router_node logic
│   └── test_eval_metrics.py      # keyword_recall, judge parsing
├── integration/             # Slower integration tests (real deps where needed)
│   ├── test_ingest_e2e.py        # Full ingest pipeline
│   ├── test_agent_e2e.py         # Full agent graph
│   └── test_eval_e2e.py          # Eval harness
└── fixtures/                # Test data
    ├── sample_claims.json        # Sample claims for testing
    └── README.md                 # Fixture documentation
```

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### Unit tests only (fast)
```bash
pytest tests/unit -m unit -v
```

### Integration tests only
```bash
pytest tests/integration -m integration -v
```

### Specific test file
```bash
pytest tests/unit/test_ingest_utils.py -v
```

### With coverage report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Skip slow tests
```bash
pytest tests/ -m "not slow" -v
```

## Test Markers

Tests are marked with markers to allow selective execution:

- `@pytest.mark.unit` — Fast unit tests with mocked dependencies
- `@pytest.mark.integration` — Slower integration tests with real/mocked external services
- `@pytest.mark.slow` — Very slow tests (e.g., actual model inference)
- `@pytest.mark.benchmark` — Long-running evaluation benchmarks

## Fixtures

Key fixtures provided by conftest.py:

**ChromaDB**
- `mock_chroma_client` — Mocked PersistentClient
- `mock_chroma_collection` — Mocked collection (add/query/get)

**Models**
- `mock_sentence_transformer` — Mocked text embedding model
- `mock_clip_model` — Mocked CLIP for image embeddings
- `mock_langchain_chat_ollama` — Mocked ChatOllama

**Data**
- `sample_claims_data` — List of test claims
- `sample_claims_file` — JSON file with test claims
- `mock_policy_results` — Mock policy search results
- `mock_damage_results` — Mock damage photo results
- `mock_claims_results` — Mock claims search results

**Parametrized**
- `insurance_type` — Parametrized: motor, home, health
- `query_family` — Parametrized: factual, cross_modal, analytical, conversational

**Utilities**
- `temp_data_dir` — Temporary directory for test files
- `spark_session` — Reusable Spark session for tests

## Best Practices

### Unit Tests
- Test single functions in isolation
- Mock all external dependencies (models, ChromaDB, Ollama)
- Keep tests fast (<100ms each)
- Use parametrization for multiple scenarios

Example:
```python
@pytest.mark.unit
def test_chunk_text_basic():
    chunks = chunk_text("a" * 1000, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]  # overlap
```

### Integration Tests
- Test full pipelines or graph execution
- Mock expensive operations (model inference, external APIs)
- Keep tests <5s each
- Use fixtures for data and mocked services

Example:
```python
@pytest.mark.integration
@patch("src.ingest.SentenceTransformer")
def test_ingest_claims_basic(mock_model_class, sample_claims_data):
    # Setup mocks and temp files
    # Call full ingest pipeline
    # Assert results
```

### Debugging

Run tests with more detail:
```bash
pytest tests/unit/test_ingest_utils.py -v -s
```

Run single test with pdb on failure:
```bash
pytest tests/unit/test_ingest_utils.py::TestChunkText::test_chunk_text_basic -v -x --pdb
```

## CI/CD Integration

Tests are automatically run on GitHub Actions on every push and pull request.

### Workflow Details

The `.github/workflows/tests.yml` workflow:
- Runs on Python 3.11, 3.12, and 3.14 (matrix strategy)
- Triggers on: push to main/develop branches, all pull requests
- Execution time: ~35-40 seconds total (parallel matrix)
- Jobs per run:
  1. **Unit tests** (~5-10s): `pytest tests/unit -m unit`
  2. **Integration tests** (~15-25s): `pytest tests/integration -m integration`
  3. **Coverage report** (~10s): `pytest tests/ --cov=src --cov-report=xml`
  4. **Codecov upload** (optional, non-blocking)

### View Results

1. Go to repo → "Actions" tab
2. Click on a workflow run
3. View logs for each job (Setup, Install, Unit Tests, Integration Tests, etc.)
4. Download coverage report artifact (30-day retention)

### Coverage Artifacts

After successful workflow runs, an HTML coverage report is available:
1. Click workflow run → "Artifacts" section
2. Download `coverage-report-<number>.zip`
3. Extract and open `htmlcov/index.html` in browser
4. View detailed coverage by file and function

### Local Testing Before Push

Test locally before pushing to catch failures early:
```bash
# Run exactly what CI will run
pytest tests/unit -m unit -v
pytest tests/integration -m integration -v

# Generate coverage report locally
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Troubleshooting Failed Runs

**PR blocked by test failure:**
1. Click "Actions" on PR
2. View failed job logs
3. Look for error in:
   - Unit test output (fast feedback)
   - Integration test output (real/mocked services)
   - Coverage report (if coverage dropped)
4. Fix locally, push again (CI runs automatically)

**Common issues:**
- `ModuleNotFoundError: No module named 'src'` → missing `pip install -e .`
- `Java not found` in Spark tests → only in local testing (CI has Java installed)
- `Connection refused` to Ollama → integration tests mock this, no real Ollama needed

### Branch Protection

(Optional) Require tests to pass before merging:
1. Go to repo Settings → Branches
2. Add rule for "main" branch
3. Enable "Require status checks to pass"
4. Select "test" workflow checks
5. Now CI must pass before PR merge

## Coverage Goals

Current coverage targets:
- **ingest.py**: 80%+ (core indexing logic)
- **agent.py**: 75%+ (node logic, excluding complex LLM interactions)
- **eval.py**: 70%+ (metrics, variant configs)

Run coverage report:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Troubleshooting

**"KeyError: 'messages'" in agent tests**
- Ensure all keys in AgentState are initialized in the test state dict

**"No module named 'src'" errors**
- Install package in development mode: `pip install -e .`
- Or ensure PYTHONPATH includes project root: `export PYTHONPATH=$PWD:$PYTHONPATH`

**Spark session errors in test_eval_e2e.py**
- Spark requires Java. Check that Java is installed: `java -version`
- Or skip Spark tests: `pytest tests/ -k "not spark"`

**Mocked LLM not called**
- Verify patch path matches import: `@patch("src.agent.llm")` not `@patch("src.ingest.llm")`
- Check that function actually uses the patched object

## Adding New Tests

1. Create test file in `tests/unit/` or `tests/integration/`
2. Use conftest fixtures or create new fixtures in conftest.py
3. Mark tests with `@pytest.mark.unit` or `@pytest.mark.integration`
4. Run: `pytest tests/path/to/test_file.py -v`
5. Check coverage: `pytest tests/path/to/test_file.py --cov=src --cov-report=term-missing`

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [PySpark testing](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_rdd.html)
