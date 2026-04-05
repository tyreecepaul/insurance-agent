"""
conftest.py
Shared pytest fixtures for all test suites.
Provides mocks for: ChromaDB, Ollama, SentenceTransformer, file system, etc.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

import numpy as np

# ── ChromaDB Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection with add/get/query methods."""
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={"ids": [], "documents": [], "distances": []})
    collection.count.return_value = 0
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Mock ChromaDB PersistentClient."""
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.list_collections.return_value = [
        MagicMock(name="policy_index"),
        MagicMock(name="damage_index"),
        MagicMock(name="claims_index"),
    ]
    return client


# ── Embedding Model Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for text embeddings."""
    model = MagicMock()
    # Return a mock embedding vector (384-dim for all-MiniLM-L6-v2)
    model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 16)  # 80-dim vector
    return model


@pytest.fixture
def mock_clip_model():
    """Mock CLIP model for image embeddings."""
    import torch
    model = MagicMock()
    preprocess = MagicMock()
    device = "cpu"
    
    # Mock encode_image to return normalized embedding
    model.encode_image.return_value = torch.ones(1, 512) / 512.0
    return model, preprocess, device


# ── Ollama/LLM Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_response():
    """Default mock response from Ollama chat endpoint."""
    return {
        "message": {
            "content": "The excess is $750 for comprehensive motor insurance."
        },
        "prompt_eval_count": 100,
        "eval_count": 30,
    }


@pytest.fixture
def mock_ollama_chat(mock_ollama_response):
    """Mock ollama.chat() function."""
    def _mock_chat(model=None, messages=None, **kwargs):
        return mock_ollama_response
    return _mock_chat


@pytest.fixture
def mock_langchain_chat_ollama():
    """Mock ChatOllama from langchain_ollama."""
    chat = MagicMock()
    chat.invoke.return_value = MagicMock(
        content="The excess is $750 for comprehensive motor insurance."
    )
    return chat


# ── Ollama Availability Check ────────────────────────────────────────────────

def ollama_available():
    """Check if Ollama service is running on localhost:11434."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


@pytest.fixture
def skip_if_no_ollama():
    """Skip test if Ollama is not available."""
    if not ollama_available():
        pytest.skip("Ollama service not running on localhost:11434")


def chroma_collections_available():
    """Check if ChromaDB has required collections initialized."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = {c.name for c in client.list_collections()}
        required = {"policy_index", "damage_index", "claims_index"}
        return required.issubset(collections)
    except:
        return False


@pytest.fixture
def skip_if_no_chroma():
    """Skip test if ChromaDB collections are not initialized."""
    if not chroma_collections_available():
        pytest.skip("ChromaDB collections not initialized. Run: python src/ingest.py")


# ── File System Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory structure for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create subdirectories
        (tmpdir / "policy_docs").mkdir()
        (tmpdir / "damage_photos").mkdir()
        (tmpdir / "claims_data").mkdir()
        
        yield tmpdir


@pytest.fixture
def sample_claims_data():
    """Sample insurance claims for testing."""
    return [
        {
            "claim_id": "CLM-2024-001",
            "claimant_name": "Sarah Chen",
            "policy_number": "POL-MOTOR-1042",
            "insurance_type": "motor",
            "incident_type": "collision",
            "incident_date": "2024-01-15",
            "claim_status": "approved",
            "claim_amount": 4050.00,
            "description": "Rear-ended at traffic lights. Vehicle sustained bumper and frame damage.",
        },
        {
            "claim_id": "CLM-2024-002",
            "claimant_name": "John Smith",
            "policy_number": "POL-HOME-5678",
            "insurance_type": "home",
            "incident_type": "water_damage",
            "incident_date": "2024-02-20",
            "claim_status": "pending",
            "claim_amount": 12500.00,
            "description": "Burst pipe in master bedroom caused water damage to ceiling and walls.",
        },
        {
            "claim_id": "CLM-2024-003",
            "claimant_name": "Alex Kim",
            "policy_number": "POL-MOTOR-9999",
            "insurance_type": "motor",
            "incident_type": "windscreen",
            "incident_date": "2024-03-10",
            "claim_status": "approved",
            "claim_amount": 350.00,
            "description": "Stone damage to windscreen. No excess applies.",
        },
    ]


@pytest.fixture
def sample_claims_file(temp_data_dir, sample_claims_data):
    """Write sample claims to a JSON file."""
    claims_file = temp_data_dir / "claims_data" / "claims.json"
    with open(claims_file, "w") as f:
        json.dump(sample_claims_data, f)
    return claims_file


# ── Retrieval Result Fixtures ────────────────────────────────────────────────

class MockRetrievalResult:
    """Mock RetrievalResult object."""
    def __init__(self, source, doc_id, content, metadata, score=0.95):
        self.source = source
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.score = score


@pytest.fixture
def mock_policy_results():
    """Mock policy search results."""
    return [
        MockRetrievalResult(
            source="policy",
            doc_id="POL-MOTOR-001_p5_c0",
            content="Comprehensive motor excess: $750 standard, $500 young drivers. Windscreen: no excess.",
            metadata={
                "insurer": "NRMA",
                "insurance_type": "motor",
                "page_number": 5,
            }
        ),
        MockRetrievalResult(
            source="policy",
            doc_id="POL-MOTOR-001_p6_c1",
            content="Exclusions: racing, rally, intentional damage, and uninsured driver liability.",
            metadata={
                "insurer": "NRMA",
                "insurance_type": "motor",
                "page_number": 6,
            }
        ),
    ]


@pytest.fixture
def mock_damage_results():
    """Mock damage photo search results."""
    return [
        MockRetrievalResult(
            source="damage",
            doc_id="car_rear_dent_01",
            content="Photo of rear bumper dent and paint damage from collision.",
            metadata={
                "damage_type": "vehicle damage",
                "insurance_category": "motor",
                "caption": "Car with significant rear bumper dent",
            }
        ),
    ]


@pytest.fixture
def mock_claims_results():
    """Mock claims search results."""
    return [
        MockRetrievalResult(
            source="claims",
            doc_id="CLM-2024-001",
            content="Claim CLM-2024-001 for Sarah Chen: motor insurance, collision incident on 2024-01-15. Status: approved. Amount: $4,050.00.",
            metadata={
                "claim_id": "CLM-2024-001",
                "claim_status": "approved",
                "insurance_type": "motor",
            }
        ),
    ]


# ── PySpark Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark_session():
    """Create a single SparkSession for all tests. Reused across tests."""
    from pyspark.sql import SparkSession
    
    spark = (
        SparkSession.builder
        .appName("InsuranceAgentTests")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    
    yield spark
    
    spark.stop()


# ── Agent State Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_agent_state():
    """Sample AgentState for testing agent nodes."""
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [
            HumanMessage(content="What is the excess for comprehensive motor insurance?")
        ],
        "query_type": "factual",
        "claim_draft": {},
        "retrieved_docs": [],
        "image_path": None,
        "detected_policy_number": None,
        "detected_insurance_type": None,
    }


# ── Eval Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_eval_result():
    """Sample EvalResult dataclass for eval tests."""
    from dataclasses import dataclass
    
    @dataclass
    class EvalResult:
        test_id: str
        family: str
        variant_key: str
        variant_name: str
        query: str
        response: str
        latency_ms: float
        input_tokens: int
        output_tokens: int
        total_tokens: int
        recall_at_5: float
        judge_score: int
        judge_reason: str
    
    return EvalResult(
        test_id="T1",
        family="factual",
        variant_key="A1",
        variant_name="baseline",
        query="What is the excess?",
        response="$750 for comprehensive motor.",
        latency_ms=150.5,
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        recall_at_5=1.0,
        judge_score=5,
        judge_reason="Correct and concise",
    )


# ── Parametrization Fixtures ────────────────────────────────────────────────

@pytest.fixture(params=["motor", "home", "health"])
def insurance_type(request):
    """Parametrized insurance type."""
    return request.param


@pytest.fixture(params=["factual", "cross_modal", "analytical", "conversational"])
def query_family(request):
    """Parametrized query family."""
    return request.param
