"""
tools.py
- Retrieval tools for LangGraph agent

Each tool queries one ChromaDB collection and returns ranked results.
Agent's retrieval planner decides which tools to invoke based on query type. 
"""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import chromadb
import numpy as np
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")

load_dotenv()

# Load configuration from config.json
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

CHROMA_DIR = _config.get("CHROMA_PERSIST_DIR", "./chroma_db")
TEXT_MODEL = _config.get("TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")
CLIP_MODEL = _config.get("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAIN = _config.get("CLIP_PRETRAINED", "openai")
TOP_K = _config.get("TOP_K", 5)

@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result from ChromaDB.
    
    Attributes:
        source: Type of document source - one of "policy", "damage", or "claims".
        doc_id: Unique identifier for the document in the collection.
        content: The actual text chunk or caption content of the document.
        metadata: Additional metadata associated with the document (e.g., insurance_type, policy_number).
        score: Relevance score from the retrieval operation (0.0 to 1.0).
    """
    source: str     # "policy" | "damage" | "claims"
    doc_id: str
    content: str    # text chunk or caption
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

# Model Cache

_text_model: Optional[SentenceTransformer] = None
_clip_model = None
_clip_preproc = None
_clip_device = None
_chroma_client: Optional[chromadb.PersistentClient] = None

def _get_text_model() -> SentenceTransformer:
    """
    Lazily load and cache the text embedding model.
    
    Returns the cached SentenceTransformer model on subsequent calls.
    Initializes the model on first call using the TEXT_MODEL constant.
    
    Returns:
        SentenceTransformer: The cached text embedding model instance.
    """
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL)
    return _text_model

def _get_clip():
    """
    Lazily load and cache the CLIP model with automatic device detection.
    
    Initializes the CLIP model (vision-language model) on first call with 
    automatic GPU/CPU device detection. Returns cached model on subsequent calls.
    Sets the model to eval mode to disable gradient computation.
    
    Returns:
        tuple: A tuple of (model, preprocess, device) containing:
            - model: The CLIP model instance
            - preprocess: Image preprocessing transform
            - device: The device ('cuda' or 'cpu') where the model is loaded
    """
    global _clip_model, _clip_preproc, _clip_device
    if _clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=device
        )
        model.eval()
        _clip_model, _clip_preproc, _clip_device = model, preprocess, device
    return _clip_model, _clip_preproc, _clip_device

def _get_chroma() -> chromadb.PersistentClient:
    """
    Lazily load and cache the Chroma vector database client.
    
    Initializes a persistent Chroma client on first call, storing embeddings 
    at the path specified by CHROMA_DIR. Returns cached client on subsequent calls.
    
    Returns:
        chromadb.PersistentClient: The cached Chroma database client instance.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client

### RRF helper

def _rrf_merge(
        dense_ids: list[str],
        bm25_ids: list[str],
        dense_docs: list[str],
        dense_metas: list[dict],
        dense_dists: list[float],
        source: str,
        k: int = 60,
    ) -> list[RetrievalResult]:
    """
    Merge and rank results from dense and BM25 retrieval using Reciprocal Rank Fusion (RRF).
    
    Combines two ranking methods (semantic similarity and keyword matching) by assigning
    a score to each document based on its rank in each list. Missing documents in either
    list receive a zero contribution from that ranking method.
    
    The RRF score formula: score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)
    where k is a constant that prevents division by zero and balances scores.
    
    Args:
        dense_ids: Document IDs from dense (semantic) retrieval in ranked order.
        bm25_ids: Document IDs from BM25 (keyword) retrieval in ranked order.
        dense_docs: Text content corresponding to dense_ids.
        dense_metas: Metadata dictionaries corresponding to dense_ids.
        dense_dists: Distance scores from dense retrieval corresponding to dense_ids.
        source: The source type to assign to results ("policy", "damage", or "claims").
        k: An integer constant to prevent division by zero (default: 60).
    
    Returns:
        list[RetrievalResult]: List of merged results sorted by RRF score (highest first).
    """
    rrf_scores: dict[str, float] = {}
 
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
 
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
 
    # Build lookup for content + metadata
    id_to_doc  = dict(zip(dense_ids, dense_docs))
    id_to_meta = dict(zip(dense_ids, dense_metas))
    id_to_dist = dict(zip(dense_ids, dense_dists))
 
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
 
    results = []
    for doc_id in sorted_ids:
        if doc_id in id_to_doc:
            results.append(RetrievalResult(
                source=source,
                doc_id=doc_id,
                content=id_to_doc[doc_id],
                metadata=id_to_meta.get(doc_id, {}),
                score=rrf_scores[doc_id],
            ))
    return results


### Policy Search
# Text Query -> Policy Clauses

def search_policy(
        query: str, 
        insurance_type: Optional[str] = None,
        top_k: int = TOP_K
    ) -> list[RetrievalResult]:
    """
    Search for insurance policy clauses matching a text query.
    
    Combines dense semantic retrieval with BM25 keyword matching, then uses
    Reciprocal Rank Fusion to merge results. Optionally filters by insurance type.
    
    Args:
        query: Text search query describing the policy information needed.
        insurance_type: Optional filter for insurance type (e.g., "auto", "home").
        top_k: Maximum number of results to return (default: TOP_K constant).
    
    Returns:
        list[RetrievalResult]: Ranked list of matching policy clauses with scores.
                               Returns error result if policy index not found.
    
    Raises:
        Returns graceful error in RetrievalResult if policy index is not found.
    """
    client = _get_chroma()
    model = _get_text_model()

    try:
        collection = client.get_collection("policy_index")
    except Exception:
        return [RetrievalResult(
            source="policy", doc_id="error", content="Policy index not found. Run: python src/ingest.py --only policy",
            score=0.0
        )]
    
    where_filter = {"insurance_type": insurance_type} if insurance_type else None
    query_embedding = model.encode(query).tolist()
 
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,          # over-fetch for RRF
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    dense_docs  = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]
    dense_dists = dense_results["distances"][0]
    dense_ids   = dense_results["ids"][0]

    # BM25 Retrieval
    # Fetch all docs (or filtered subset) for BM25 scoring
    all_data = collection.get(where=where_filter, include=["documents", "metadatas"])
    corpus    = all_data["documents"]
    all_ids   = all_data["ids"]

    bm25_ranked_ids: list[str] = []
    if corpus:
        tokenised = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenised)
        bm25_scores = bm25.get_scores(query.lower().split())
        ranked_idx  = np.argsort(bm25_scores)[::-1][: top_k * 2]
        bm25_ranked_ids = [all_ids[i] for i in ranked_idx]

    # reciprocal rank fusion
    results = _rrf_merge(
        dense_ids, bm25_ranked_ids,
        dense_docs, dense_metas, dense_dists,
        source="policy",
    )
    return results[:top_k]

### Damage Assessment
# Image -> Similar Damage and Policy Link

def search_damage(
        image_path: Optional[str] = None,
        text_query: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> list[RetrievalResult]:
    """
    Search for similar damage assessments using image or text query.
    
    Uses CLIP (vision-language model) to encode either an image or text description
    into embeddings, then finds similar damage photos/captions in the database.
    
    Args:
        image_path: Path to a damage photo to find similar cases. Mutually exclusive with text_query.
        text_query: Text description of damage to find similar cases. Mutually exclusive with image_path.
        top_k: Maximum number of similar damage records to return (default: TOP_K constant).
    
    Returns:
        list[RetrievalResult]: Ranked list of similar damage assessments with scores (0.0-1.0).
                               Returns error result if damage index not found.
    
    Raises:
        ValueError: If neither image_path nor text_query is provided.
        Returns graceful error in RetrievalResult if damage index is not found.
    """
    client = _get_chroma()
    clip_model, preprocess, device = _get_clip()
 
    try:
        collection = client.get_collection("damage_index")
    except Exception:
        return [RetrievalResult(
            source="damage", doc_id="error",
            content="Damage index not found. Run: python src/ingest.py --only damage",
            score=0.0
        )]
    
    # Build query embedding
    if image_path:
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(img)
    elif text_query:
        import open_clip as oc
        tokeniser = oc.get_tokenizer(CLIP_MODEL)
        tokens = tokeniser([text_query]).to(device)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
    else:
        raise ValueError("Provide either image_path or text_query")

    features = features / features.norm(dim=-1, keepdim=True)
    embedding = features.squeeze().cpu().numpy().tolist()
 
    # Query ChromaDB
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
 
    results = []
    for doc_id, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append(RetrievalResult(
            source="damage",
            doc_id=doc_id,
            content=doc,
            metadata=meta,
            score=float(1 - dist),
        ))
    return results


### Claims Lookup
# Semantic and Structured Filter

def search_claims(
        query: str,
        policy_number: Optional[str] = None,
        claim_status: Optional[str] = None,
        insurance_type: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> list[RetrievalResult]:
    """
    Search insurance claims using semantic query with optional structured filters.
    
    Performs semantic search on claim records using text embeddings, with support
    for filtering by policy number, claim status, and insurance type.
    
    Args:
        query: Text search query (e.g., "car accident claim" or "water damage").
        policy_number: Optional filter for specific policy number.
        claim_status: Optional filter for claim status (e.g., "approved", "rejected", "pending").
        insurance_type: Optional filter for insurance type (e.g., "auto", "home").
        top_k: Maximum number of claims to return (default: TOP_K constant).
    
    Returns:
        list[RetrievalResult]: Ranked list of matching claims with scores.
                               Returns error result if claims index not found.
    
    Raises:
        Returns graceful error in RetrievalResult if claims index is not found.
    """

    client = _get_chroma()
    model  = _get_text_model()
 
    try:
        collection = client.get_collection("claims_index")
    except Exception:
        return [RetrievalResult(
            source="claims", doc_id="error",
            content="Claims index not found. Run: python src/ingest.py --only claims",
            score=0.0
        )]
    
    # Build Metadata filter
    where_conditions = []
    if policy_number:
        where_conditions.append({"policy_number": {"$eq": policy_number}})
    if claim_status:
        where_conditions.append({"claim_status": {"$eq": claim_status}})
    if insurance_type:
        where_conditions.append({"insurance_type": {"$eq": insurance_type}})
 
    where_filter = (
        {"$and": where_conditions} if len(where_conditions) > 1
        else where_conditions[0] if where_conditions
        else None
    )

    # Semantic Query
    embedding = model.encode(query).tolist()
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for doc_id, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append(RetrievalResult(
            source="claims",
            doc_id=doc_id,
            content=doc,
            metadata=meta,
            score=float(1 - dist),
        ))
    return results

