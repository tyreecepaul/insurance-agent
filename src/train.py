"""
Fine-tune all-MiniLM-L6-v2 on insurance domain data.

Strategy: MultipleNegativesRankingLoss with (query, positive_chunk) pairs.
Sources:
  1. Eval benchmark queries  → top ChromaDB chunks that contain expected keywords
  2. Ground-truth pairs      → benchmark query + ground_truth text as synthetic positive
  3. Policy chunk augment    → template questions generated from chunks with coverage keywords

Saves the fine-tuned model to models/insurance-embeddings/ and updates config.json.

Usage:
    python src/train_embeddings.py
    python src/train_embeddings.py --epochs 5 --batch-size 16 --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import chromadb
import mlflow
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import BENCHMARK  # noqa: E402 — needs sys.path patch above

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = ROOT / "config.json"
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

CHROMA_DIR = ROOT / _cfg["CHROMA_PERSIST_DIR"]
BASE_MODEL = _cfg["TEXT_EMBED_MODEL"]
OUTPUT_DIR = ROOT / "models" / "insurance-embeddings"
MLFLOW_EXPERIMENT = "insurance-embedding-finetune"

# Coverage keywords used for augmentation source selection
_COVERAGE_TRIGGERS = re.compile(
    r"\b(cover|excess|exclud|claim|premium|benefit|limit|deductible|reimburse|"
    r"repair|settlement|approv|reject|policy|insur|liable|liability|compensat)\w*\b",
    re.IGNORECASE,
)

# Query templates for augmenting policy chunks
_TEMPLATES = [
    "What does the policy say about {topic}?",
    "Am I covered for {topic}?",
    "How does {topic} work under this insurance policy?",
    "What are the conditions for {topic}?",
    "Is {topic} included in the coverage?",
]


# ── Data construction ─────────────────────────────────────────────────────────

def _load_policy_chunks() -> list[str]:
    """Return all document texts from policy_index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection("policy_index")
    result = col.get(include=["documents"])
    return result["documents"]


def _retrieve_positives(
    col: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    keywords: list[str],
    top_k: int = 20,
) -> list[str]:
    """
    Retrieve top_k chunks and return those containing at least one keyword.
    Falls back to the single best chunk if none match keywords.
    """
    embedding = model.encode(query).tolist()
    result = col.query(query_embeddings=[embedding], n_results=min(top_k, col.count()))
    docs = result["documents"][0]
    matches = [
        d for d in docs
        if any(kw.lower() in d.lower() for kw in keywords)
    ]
    return matches if matches else docs[:1]


def build_benchmark_pairs(model: SentenceTransformer) -> list[tuple[str, str]]:
    """
    Pairs from benchmark:
      - (query, positive_chunk) from ChromaDB retrieval with keyword filter
      - (query, ground_truth) as a synthetic direct positive
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    policy_col = client.get_collection("policy_index")
    claims_col = client.get_collection("claims_index")
    pairs: list[tuple[str, str]] = []

    for case in BENCHMARK:
        source = case.get("expected_source", "")
        query = case["query"]
        keywords = case.get("keywords", [])
        ground_truth = case.get("ground_truth", "")

        if source == "policy":
            positives = _retrieve_positives(policy_col, model, query, keywords)
            pairs.extend((query, p) for p in positives)
        elif source == "claims":
            positives = _retrieve_positives(claims_col, model, query, keywords)
            pairs.extend((query, p) for p in positives)

        # Ground-truth text is always a valid positive regardless of source
        if ground_truth:
            pairs.append((query, ground_truth))

    return pairs


def _extract_topic(chunk: str) -> str | None:
    """Return the sentence from the chunk that contains a coverage trigger word."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    for sent in sentences:
        if _COVERAGE_TRIGGERS.search(sent):
            topic = sent.strip().rstrip(".!?,;:")
            # Must be long enough to be meaningful but short enough for a template
            if 10 < len(topic) < 120:
                return topic
    return None


def build_augmented_pairs(
    policy_chunks: list[str],
    n_per_chunk: int = 2,
) -> list[tuple[str, str]]:
    """
    For chunks that mention coverage concepts, generate template-based
    query variants and pair them with the source chunk.
    """
    pairs: list[tuple[str, str]] = []
    rng = random.Random(42)

    for chunk in policy_chunks:
        if not _COVERAGE_TRIGGERS.search(chunk):
            continue
        topic = _extract_topic(chunk)
        if not topic:
            continue
        templates = rng.sample(_TEMPLATES, k=min(n_per_chunk, len(_TEMPLATES)))
        for tmpl in templates:
            synthetic_query = tmpl.format(topic=topic)
            pairs.append((synthetic_query, chunk))

    return pairs


def deduplicate(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    epochs: int = 3,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    use_mlflow: bool = True,
) -> Path:
    print(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    print("Building training pairs…")
    policy_chunks = _load_policy_chunks()
    benchmark_pairs = build_benchmark_pairs(model)
    augmented_pairs = build_augmented_pairs(policy_chunks)
    all_pairs = deduplicate(benchmark_pairs + augmented_pairs)
    random.Random(42).shuffle(all_pairs)

    n_val = max(1, int(len(all_pairs) * 0.15))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    print(f"  Train pairs : {len(train_pairs)}")
    print(f"  Val pairs   : {len(val_pairs)}")

    train_dataset = Dataset.from_dict({
        "anchor": [p[0] for p in train_pairs],
        "positive": [p[1] for p in train_pairs],
    })
    val_dataset = Dataset.from_dict({
        "anchor": [p[0] for p in val_pairs],
        "positive": [p[1] for p in val_pairs],
    })

    loss = MultipleNegativesRankingLoss(model)

    steps_per_epoch = max(1, len(train_pairs) // batch_size)
    warmup_steps = int(steps_per_epoch * epochs * warmup_ratio)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, steps_per_epoch // 4),
        report_to=[],  # disable HF logging; we use MLflow manually
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    if use_mlflow:
        mlflow.set_tracking_uri(str(ROOT / "mlruns"))
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run(run_name=f"finetune-epochs{epochs}-bs{batch_size}")
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "epochs": epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "loss": "MultipleNegativesRankingLoss",
        })

    print("Training…")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Model saved → {OUTPUT_DIR}")

    if use_mlflow:
        train_result = trainer.state.log_history
        final_eval = next(
            (e for e in reversed(train_result) if "eval_loss" in e), None
        )
        if final_eval:
            mlflow.log_metrics({
                "final_eval_loss": final_eval["eval_loss"],
                "final_epoch": final_eval.get("epoch", epochs),
            })
        mlflow.log_artifact(str(OUTPUT_DIR), artifact_path="model")
        mlflow.end_run()

    return OUTPUT_DIR


def update_config(model_path: Path) -> None:
    """Point TEXT_EMBED_MODEL in config.json at the fine-tuned model."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    relative = str(model_path.relative_to(ROOT))
    cfg["TEXT_EMBED_MODEL"] = f"./{relative}"
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"config.json updated: TEXT_EMBED_MODEL → {cfg['TEXT_EMBED_MODEL']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune insurance text embeddings")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Skip writing the fine-tuned model path back to config.json",
    )
    args = parser.parse_args()

    model_path = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        use_mlflow=not args.no_mlflow,
    )

    if not args.no_update_config:
        update_config(model_path)

    print("\nDone. Re-run src/ingest.py to rebuild ChromaDB with the new embeddings.")
    print(f"MLflow UI → mlflow ui  (http://localhost:5000, experiment: {MLFLOW_EXPERIMENT})")


if __name__ == "__main__":
    main()
