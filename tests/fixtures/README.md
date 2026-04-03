# Test Fixtures

This directory contains sample data for testing the insurance agent.

## Contents

### sample_claims.json
Sample insurance claims data with various statuses and types.

**Records:**
- CLM-2024-001: Motor collision (approved)
- CLM-2024-002: Home water damage (pending)
- CLM-2024-003: Motor windscreen (approved)
- CLM-2024-004: Home water damage (rejected - gradual deterioration)
- CLM-2023-087: Motor collision (approved)

**Usage in tests:**
```python
import json
from pathlib import Path

with open("tests/fixtures/sample_claims.json") as f:
    claims = json.load(f)
```

## How to Add More Fixtures

### PDF Test Files
For testing ingest.py PDF parsing, you can use:
1. Create dummy PDFs with `fitz` or use real sample PDFs
2. Place in `tests/fixtures/sample_pdfs/`
3. Mock PDF content in conftest.py if actual files not needed

### Image Test Files
For testing CLIP embeddings:
1. Use small test images (PNG/JPG)
2. Place in `tests/fixtures/sample_images/`
3. Mock CLIP embeddings in conftest.py for unit tests
4. Use real images in integration tests if needed

### Custom Claim Data
Edit sample_claims.json to add more test cases or scenarios.

## Testing Best Practices

- **Unit tests**: Use conftest.py fixtures and mocks (fast, <1s per test)
- **Integration tests**: Use actual files from this directory if needed
- **Mocking**: Prefer mocking expensive operations (PDF parsing, image embedding, LLM calls)

## File Sizes

Keep fixtures small to keep test suite fast:
- Claims JSON: <10KB
- Sample PDFs: 50-100KB each (minimal content)
- Sample images: 10-50KB each (small dimensions)
