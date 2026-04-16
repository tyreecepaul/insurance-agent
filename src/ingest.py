"""
ingest.py
- Build all 3 ChromaDB indices from raw data

Run once before starting agent:
    python src/ingest.py

Or selectively:
    python src/ingest.py --only policy
    python src/ingest.py --only damage
    python src/ingest.py --only claims
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
 
import fitz  # pymupdf
import chromadb
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import open_clip
import torch

load_dotenv()

# Load configuration from config.json (graceful fallback if missing)
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
try:
    with open(CONFIG_PATH, "r") as f:
        _config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # FileNotFoundError: config.json does not exist (expected in new installs).
    # JSONDecodeError: config.json exists but is malformed (missing or invalid JSON).
    _config = {}

CHROMA_DIR = _config.get("CHROMA_PERSIST_DIR", "./chroma_db")
TEXT_MODEL = _config.get("TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")
CLIP_MODEL = _config.get("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAIN = _config.get("CLIP_PRETRAINED", "openai")
CHUNK_SIZE = _config.get("CHUNK_SIZE", 500)   # tokens approx (characters / 4)
CHUNK_OVERLAP = _config.get("CHUNK_OVERLAP", 50)
 
POLICY_DIR = Path(_config.get("POLICY_DIR", "data/policy_docs"))
DAMAGE_DIR = Path(_config.get("DAMAGE_DIR", "data/damage_photos"))
CLAIMS_FILE = Path(_config.get("CLAIMS_FILE", "data/claims_data/claims.json"))

# Module-level cache for BLIP-2 model to avoid reloading on every generate_caption call.
# Stores tuple (processor, model, device) or None if unavailable.
_BLIP2 = None

def get_client() -> chromadb.PersistentClient:
    """
    Get or create a ChromaDB persistent client.
    
    Returns a persistent client pointing to the database directory configured
    in CHROMA_DIR. Reuse the same client instance for multiple operations.
    
    Returns:
        chromadb.PersistentClient: A ChromaDB client for accessing vector collections.
    """
    return chromadb.PersistentClient(path=CHROMA_DIR)


### Policy Index
# Text chunks from PDFs

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for vector indexing.
    
    Divides text into fixed-size chunks with specified overlap to preserve
    context across chunk boundaries. Used for embedding and indexing long documents.
    
    Args:
        text: The text to chunk (e.g., a PDF page).
        chunk_size: Characters per chunk (default: CHUNK_SIZE from config).
        overlap: Characters of overlap between consecutive chunks (default: CHUNK_OVERLAP from config).
    
    Returns:
        list[str]: List of text chunks with the specified overlap between them.
    """
    # MEDIUM fix: step <= 0 causes an infinite loop; reject invalid configs early.
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size}); "
            f"got step={step} which would cause an infinite loop."
        )
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks

def ingest_policies(client: chromadb.PersistentClient, model: SentenceTransformer):
    """
    Index all policy PDFs into ChromaDB collection.
    
    Extracts text from PDF files in POLICY_DIR, chunks the text, generates
    embeddings using the provided model, and stores them in the "policy_index"
    collection. Automatically detects insurer and insurance type from filename.
    Skips documents that have already been indexed.
    
    Args:
        client: ChromaDB client for database access.
        model: SentenceTransformer model for generating text embeddings.
    
    Returns:
        None. Prints status messages and updates the policy_index collection.
    """
    collection = client.get_or_create_collection(
        name="policy_index",
        metadata={"hnsw:space": "cosine"},
    )
    existing = set(collection.get()["ids"])
 
    pdf_files = list(POLICY_DIR.glob("*.pdf"))
    if not pdf_files:
        print("   No PDFs found in data/policy_docs/ — add policy PDFs and re-run.")
        print("   Download sample PDFs from: NRMA, Allianz, RACQ, Medibank websites.")
        return
    
    print(f"\n Indexing {len(pdf_files)} policy PDFs...")

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        doc = fitz.open(pdf_path)
        policy_id = pdf_path.stem                   # filename without extension
        insurer   = _guess_insurer(pdf_path.name)
        ins_type  = _guess_insurance_type(pdf_path.name)
 
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if not page_text.strip():
                continue
 
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{policy_id}_p{page_num}_c{chunk_idx}"
                if doc_id in existing:
                    continue
                    
                embedding = model.encode(chunk).tolist()
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "policy_id":       policy_id,
                        "insurer":         insurer,
                        "insurance_type":  ins_type,
                        "page_number":     page_num,
                        "chunk_index":     chunk_idx,
                        "source_file":     pdf_path.name,
                    }],
                )

    print(f"  Policy index: {collection.count()} chunks stored.")

def _guess_insurer(filename: str) -> str:
    """
    Infer insurance company from filename.
    
    Matches against a list of common Australian insurers (NRMA, Allianz, RACQ, etc.).
    
    Args:
        filename: The filename or path to check.
    
    Returns:
        str: Insurer name if found in filename, otherwise "unknown".
    """
    fn = filename.lower()
    for name in ["nrma", "allianz", "racq", "medibank", "bupa", "nib", "suncorp", "gio"]:
        if name in fn:
            return name
    return "unknown"

def _guess_insurance_type(filename: str) -> str:
    """
    Infer insurance policy type from filename.
    
    Matches keywords in the filename to categorize the insurance type
    (motor, home, health, or general).
    
    Args:
        filename: The filename or path to check.
    
    Returns:
        str: Insurance type ("motor", "home", "health") or "general" if no match.
    """
    fn = filename.lower()
    if any(w in fn for w in ["motor", "car", "vehicle", "auto"]):
        return "motor"
    if any(w in fn for w in ["home", "house", "property", "building"]):
        return "home"
    if any(w in fn for w in ["health", "medical", "hospital"]):
        return "health"
    return "general"

### Damage Index 
# Images via CLIP + BLIP-2 captions

def load_clip(model_name: str, pretrained: str):
    """
    Load a CLIP model with automatic device detection.
    
    Creates a vision-language model from OpenAI's CLIP, automatically selecting
    GPU (CUDA) if available, otherwise falls back to CPU. Sets model to eval mode.
    
    Args:
        model_name: CLIP model identifier (e.g., "ViT-B-32").
        pretrained: Pretrained weights source (e.g., "openai").
    
    Returns:
        tuple: (model, preprocess, device) containing:
            - model: The CLIP model instance in eval mode
            - preprocess: Image preprocessing transform function
            - device: The device ('cuda' or 'cpu') where the model is loaded
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess, device

def _get_blip2():
    """
    Lazy-load BLIP-2 once and cache it.
    
    Loads the processor and model on first call and caches the result globally.
    Subsequent calls return the cached tuple. Device selection is centralized here.
    
    Returns:
        tuple: (processor, model, device) if successful, None if unavailable.
    """
    global _BLIP2
    if _BLIP2 is not None:
        return _BLIP2

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-sm")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-sm",
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model = blip_model.to(device)
        _BLIP2 = (processor, blip_model, device)
        return _BLIP2
    except Exception as e:
        print(f"   BLIP-2 unavailable ({e}), captions will use filename fallback.")
        _BLIP2 = None
        return None


def generate_caption(image_path: Path) -> str:
    """
    Generate a natural language caption for an insurance damage photo.
    
    Uses the cached BLIP-2 model (loaded once on first call via _get_blip2).
    Falls back to filename-based caption if BLIP-2 is unavailable or if
    caption generation fails.

    Args:
        image_path: Path to the damage photo image file.

    Returns:
        str: A natural language description of the damage image.
    """
    blip2 = _get_blip2()
    if blip2 is None:
        # BLIP-2 unavailable — use filename-based fallback
        stem = image_path.stem.replace("_", " ").replace("-", " ")
        return f"insurance damage photo: {stem}"

    processor, blip_model, device = blip2

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            images=image,
            text="Describe the damage in this insurance photo:",
            return_tensors="pt",
        ).to(device, torch.float16)

        output = blip_model.generate(**inputs, max_new_tokens=80)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    except Exception as e:
        # Caption generation failed — fall back to filename
        stem = image_path.stem.replace("_", " ").replace("-", " ")
        print(f"   Caption generation failed ({e}), using filename as caption.")
        return f"insurance damage photo: {stem}"
    
def embed_image_clip(image_path: Path, clip_model, preprocess, device) -> list[float]:
    """
    Generate a CLIP embedding for an image.
    
    Loads the image, preprocesses it, passes through CLIP vision encoder, and
    returns the normalized embedding vector suitable for vector similarity search.
    
    Args:
        image_path: Path to the image file.
        clip_model: Loaded CLIP model instance.
        preprocess: Image preprocessing transform function from CLIP.
        device: Device ('cuda' or 'cpu') to run inference on.
    
    Returns:
        list[float]: Normalized embedding vector as a list of floats.
    """
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().tolist()

def ingest_damage_photos(client: chromadb.PersistentClient):
    """
    Index all damage photos into ChromaDB collection.
    
    Processes all images in DAMAGE_DIR by:
      1. Generating CLIP embeddings for vision-based similarity search
      2. Creating natural language captions using BLIP-2 (or filename fallback)
      3. Classifying damage type and insurance category from filename
      4. Storing results in the "damage_index" ChromaDB collection
    
    Skips images that have already been indexed.
    
    Args:
        client: ChromaDB client for database access.
    
    Returns:
        None. Prints status messages and updates the damage_index collection.
    """
    collection = client.get_or_create_collection(
        name="damage_index",
        metadata={"hnsw:space": "cosine"},
    )
    existing = set(collection.get()["ids"])
 
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in DAMAGE_DIR.iterdir() if p.suffix.lower() in img_extensions]
 
    if not images:
        print("   No images found in data/damage_photos/ — add damage photos and re-run.")
        print("   Tip: download sample damage photos from Unsplash or use your own.")
        return
 
    print(f"\n   Indexing {len(images)} damage photos...")
    clip_model, preprocess, device = load_clip(CLIP_MODEL, CLIP_PRETRAIN)

    # BLIP-2 is loaded once on first call to generate_caption via the cached _get_blip2().
    # No need to pre-load here; the cache ensures "load once, reuse" behavior.

    for img_path in tqdm(images, desc="Images"):
        doc_id = img_path.stem
        if doc_id in existing:
            continue

        # Embed with CLIP
        embedding = embed_image_clip(img_path, clip_model, preprocess, device)

        # Generate caption using cached BLIP-2 loader
        caption = generate_caption(img_path)
        
        # Clear GPU cache after each image to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        # Infer damage type from filename convention:
        # e.g. "car_rear_dent_01.jpg" → damage_type="car", severity="medium"
        damage_type, insurance_category = _classify_damage_from_filename(img_path.stem)
 
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[caption],
            metadatas=[{
                "image_path":         str(img_path),
                "caption":            caption,
                "damage_type":        damage_type,
                "insurance_category": insurance_category,
                "filename":           img_path.name,
            }],
        )
 
    print(f"  Damage index: {collection.count()} images stored.")

def _classify_damage_from_filename(stem: str) -> tuple[str, str]:
    """
    Infer damage type and insurance category from filename conventions.
    
    Parses filename keywords to classify the damage (e.g., "car_dent", "roof_damage")
    and assigns an appropriate insurance category (motor, home, or general).
    
    Examples:
        - "car_rear_dent_01" → ("vehicle damage", "motor")
        - "roof_hail_damage" → ("weather damage", "home")
        - "flood_basement" → ("weather damage", "home")
    
    Args:
        stem: Filename stem (without extension) to classify.
    
    Returns:
        tuple: (damage_type, insurance_category) where both are strings.
    """
    stem = stem.lower()
    if any(w in stem for w in ["car", "vehicle", "bumper", "dent", "scratch", "windscreen", "crash"]):
        return "vehicle damage", "motor"
    if any(w in stem for w in ["roof", "ceiling", "wall", "floor", "window", "house", "home"]):
        return "property damage", "home"
    if any(w in stem for w in ["flood", "water", "storm", "hail"]):
        return "weather damage", "home"
    if any(w in stem for w in ["fire", "smoke", "burn"]):
        return "fire damage", "home"
    return "general damage", "general"


### Claims Index
# Structured metadata from JSON

def ingest_claims(client: chromadb.PersistentClient, model: SentenceTransformer):
    """
    Index all insurance claims records into ChromaDB collection.
    
    Loads claim records from CLAIMS_FILE (JSON), generates semantic embeddings
    using the provided text model, and stores them in the "claims_index" collection.
    All claim metadata (policy_number, claim_status, insurance_type, etc.) is
    preserved for filtering and structured queries.
    
    Skips claims that have already been indexed.
    
    Args:
        client: ChromaDB client for database access.
        model: SentenceTransformer model for generating text embeddings.
    
    Returns:
        None. Prints status messages and updates the claims_index collection.
    """
    collection = client.get_or_create_collection(
        name="claims_index",
        metadata={"hnsw:space": "cosine"},
    )
    existing = set(collection.get()["ids"])
 
    if not CLAIMS_FILE.exists():
        print("   claims.json not found at data/claims_data/claims.json")
        return
 
    print("\n  Indexing claims records...")
 
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)
 
    for claim in tqdm(claims, desc="Claims"):
        cid = claim["claim_id"]
        if cid in existing:
            continue
 
        # Build a natural language summary for semantic search
        summary = (
            f"Claim {cid} for {claim['claimant_name']}: "
            f"{claim['insurance_type']} insurance, {claim['incident_type']} incident "
            f"on {claim['incident_date']}. Status: {claim['claim_status']}. "
            f"Amount: ${claim['claim_amount']:,.2f}. "
            f"{claim['description']}"
        )
 
        embedding = model.encode(summary).tolist()
 
        collection.add(
            ids=[cid],
            embeddings=[embedding],
            documents=[summary],
            metadatas={
                k: (str(v) if v is not None else "")
                for k, v in claim.items()
            },
        )
 
    print(f"  Claims index: {collection.count()} records stored.")


### MAIN    

def main(only: Optional[str] = None):
    """
    Main entry point for building all ChromaDB indices.
    
    Orchestrates indexing of policy PDFs, damage photos, and insurance claims.
    Loads the shared text embedding model once and passes it to both policy and
    claims indexing functions. Can optionally index only one collection type.
    
    Args:
        only: Optional filter to index only one collection type:
              - None: Index all three collections (default)
              - "policy": Index only policies
              - "damage": Index only damage photos
              - "claims": Index only claims
    
    Returns:
        None. Creates/updates ChromaDB collections on disk and prints status messages.
    """
    client = get_client()
 
    # Load text embedding model (shared by policy + claims)
    print(f"Loading text embedding model: {TEXT_MODEL}...")
    text_model = SentenceTransformer(TEXT_MODEL)
 
    if only in (None, "policy"):
        ingest_policies(client, text_model)
 
    if only in (None, "damage"):
        ingest_damage_photos(client)
 
    if only in (None, "claims"):
        ingest_claims(client, text_model)
 
    print("\n  All indices ready. You can now run: python main.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["policy", "damage", "claims"],
                        help="Ingest only one index")
    args = parser.parse_args()
    main(only=args.only)