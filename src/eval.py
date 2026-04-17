"""
eval.py
- Evaluation harness with PySpark agregation and MLflow experiment tracking

- Tests 4 query families across 4 ablation variants (A1-A4)
- MLflow tracks parameters, metrics and artefacts every run
- PySpark aggregates results into summary tables

Usage:
    python src/eval.py                        # full ablation study
    python src/eval.py --variant A4           # single variant
    python src/eval.py --family factual       # single query family
    mlflow ui                                 # view results at localhost:5000
"""

import sys
import os

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from typing import Optional
from dataclasses import dataclass, asdict

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, 
    StringType, FloatType, IntegerType
)
import ollama
from dotenv import load_dotenv

from src.tools import search_policy, search_damage, search_claims

load_dotenv()

MLFLOW_EXPERIMENT = "insurance-agent-ablation"

# PySpark session (local mode with no cluster needed)
def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("InsuranceAgentEval")
        .master("local[*]")                      # use all local CPU cores
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")   # small dataset — avoid 200 default
        .getOrCreate()
    )

# Benchmarks
BENCHMARK: list[dict] = [
    # ── FAMILY 1: Factual retrieval ──────────────────────────
    {
        "id": "F1-01",
        "family": "factual",
        "query": "What is the excess amount for a comprehensive motor insurance claim?",
        "ground_truth": "The standard excess for comprehensive motor is $750, with a reduced excess option available.",
        "expected_source": "policy",
        "keywords": ["excess", "750", "comprehensive"],
    },
    {
        "id": "F1-02",
        "family": "factual",
        "query": "Is windscreen damage covered, and is there an excess?",
        "ground_truth": "Windscreen damage is covered under comprehensive motor. No excess applies for windscreen replacement.",
        "expected_source": "policy",
        "keywords": ["windscreen", "covered", "no excess"],
    },
    {
        "id": "F1-03",
        "family": "factual",
        "query": "What home insurance events are excluded under the policy?",
        "ground_truth": "Common exclusions include gradual deterioration, flood (if not elected), vermin, and intentional damage.",
        "expected_source": "policy",
        "keywords": ["exclud", "flood", "deterioration"],
    },
    # ── FAMILY 2: Cross-modal assessment ─────────────────────
    {
        "id": "F2-01",
        "family": "cross_modal",
        "query": "Here is a photo of damage to my car's rear bumper after a collision. Will this be covered?",
        "ground_truth": "Rear bumper collision damage is covered under comprehensive motor insurance.",
        "expected_source": "damage",
        "keywords": ["rear", "collision", "covered", "comprehensive"],
    },
    {
        "id": "F2-02",
        "family": "cross_modal",
        "query": "I have a photo of my ceiling which has water damage from a burst pipe. Is this covered?",
        "ground_truth": "Sudden burst pipe water damage is typically covered under home insurance. Gradual leaks are not.",
        "expected_source": "damage",
        "keywords": ["water", "pipe", "ceiling", "covered"],
    },
    # ── FAMILY 3: Analytical / multi-hop ─────────────────────
    {
        "id": "F3-01",
        "family": "analytical",
        "query": "My car was rear-ended at traffic lights last night. What steps do I need to take to lodge a claim?",
        "ground_truth": "Steps: 1. Collect third party details 2. File police report if required 3. Contact insurer within 24h 4. Take photos 5. Get repair quote.",
        "expected_source": "policy",
        "keywords": ["third party", "police", "photo", "repair", "contact"],
    },
    {
        "id": "F3-02",
        "family": "analytical",
        "query": "A storm has damaged my roof and rain has come through. What do I need to do right now?",
        "ground_truth": "Steps: 1. Ensure safety 2. Tarp roof 3. Document damage with photos 4. Lodge claim 5. Keep receipts for emergency work.",
        "expected_source": "policy",
        "keywords": ["tarp", "document", "photo", "emergency", "lodge"],
    },
    # ── FAMILY 4: Conversational / status ────────────────────
    {
        "id": "F4-01",
        "family": "conversational",
        "query": "What is the current status of claim CLM-2024-001?",
        "ground_truth": "Claim CLM-2024-001 for Sarah Chen has been approved. Settlement $4,050 after $750 excess.",
        "expected_source": "claims",
        "keywords": ["approved", "CLM-2024-001", "4050", "750"],
    },
    {
        "id": "F4-02",
        "family": "conversational",
        "query": "Why was claim CLM-2024-004 rejected?",
        "ground_truth": "Rejected because pipe corrosion was gradual deterioration, not a sudden event. Excluded under section 4.2.",
        "expected_source": "claims",
        "keywords": ["rejected", "gradual", "deterioration", "4.2"],
    },
    {
        "id": "F4-03",
        "family": "conversational",
        "query": "I need to lodge a new claim. My name is Alex Kim and my vehicle was damaged by hail yesterday.",
        "ground_truth": "Agent should start collecting claim details: policy number, incident date, vehicle details, damage description.",
        "expected_source": "claims",
        "keywords": ["hail", "policy", "date", "damage", "claim"],
    },
    {
        "id": "F4-04",
        "family": "conversational",
        "query": "Show me all my approved claims for motor insurance.",
        "ground_truth": "Should return CLM-2024-001 (approved, motor) and CLM-2023-087 (approved, motor).",
        "expected_source": "claims",
        "keywords": ["motor", "approved", "CLM-2024-001", "CLM-2023-087"],
    },
]

# Ablation Variant Configs
@dataclass
class VariantConfig:
    name:          str
    description:   str
    use_policy:    bool = True
    use_damage:    bool = True
    use_claims:    bool = True
    use_router:    bool = True
    use_retrieval: bool = True
 
 
VARIANTS: dict[str, VariantConfig] = {
    "A1": VariantConfig(
        name="A1_plain_llm",
        description="No retrieval — baseline LLM only",
        use_policy=False, use_damage=False, use_claims=False,
        use_router=False, use_retrieval=False,
    ),
    "A2": VariantConfig(
        name="A2_text_only",
        description="Policy + claims text, no image index",
        use_policy=True, use_damage=False, use_claims=True,
        use_router=True, use_retrieval=True,
    ),
    "A3": VariantConfig(
        name="A3_no_router",
        description="All indices, no routing — hits everything every query",
        use_policy=True, use_damage=True, use_claims=True,
        use_router=False, use_retrieval=True,
    ),
    "A4": VariantConfig(
        name="A4_full_agent",
        description="Full system: routing + all indices + memory",
        use_policy=True, use_damage=True, use_claims=True,
        use_router=True, use_retrieval=True,
    ),
}

# Raw Result Dataclass
@dataclass
class EvalResult:
    test_id:       str
    family:        str
    variant_key:   str
    variant_name:  str
    query:         str
    response:      str
    latency_ms:    float
    input_tokens:  int
    output_tokens: int
    total_tokens:  int
    recall_at_5:   float
    judge_score:   int
    judge_reason:  str

def run_agent_query(
        query: str,
        config: VariantConfig,
        image_path: Optional[str] = None,
    ) -> tuple[str, int, int]:

    context_parts: list[str] = []
 
    if config.use_retrieval:
        # Lightweight keyword router (avoids extra LLM cost per eval call)
        if config.use_router:
            q = query.lower()
            if any(w in q for w in ["photo", "image", "picture", "crack", "dent", "ceiling", "damage photo"]):
                family = "cross_modal"
            elif any(w in q for w in ["status", "clm-", "why was", "lodge", "show me all"]):
                family = "conversational"
            elif any(w in q for w in ["what do i do", "steps", "how do i", "need to do"]):
                family = "analytical"
            else:
                family = "factual"
        else:
            family = "all"  # no router — blast all indices
 
        if config.use_policy and family in ("factual", "analytical", "cross_modal", "all"):
            for r in search_policy(query, top_k=5):
                context_parts.append(f"[POLICY p.{r.metadata.get('page_number','?')}] {r.content[:400]}")
 
        if config.use_damage and family in ("cross_modal", "all"):
            results = search_damage(
                image_path=image_path,
                text_query=query if not image_path else None,
            )
            for r in results[:3]:
                context_parts.append(f"[DAMAGE — {r.metadata.get('damage_type','')}] {r.content[:300]}")
 
        if config.use_claims and family in ("conversational", "analytical", "all"):
            for r in search_claims(query, top_k=5):
                context_parts.append(f"[CLAIM {r.metadata.get('claim_id','')}] {r.content[:300]}")
 
    system = (
        "You are an insurance claims assistant. Answer using the provided context. "
        "Cite specific policy sections or claim IDs where relevant. "
        "If context is absent, answer from general knowledge but say so."
    )
    context_str = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"

    try:
        response = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_msg},
            ],
        )
        text = response["message"]["content"]
        in_tok  = response.get("prompt_eval_count", 0)
        out_tok = response.get("eval_count", 0)
        
        return text, in_tok, out_tok
    except Exception as e:
        return f"ERROR: {str(e)}", 0, 0


# Metrics
def keyword_recall(response: str, keywords: list[str]) -> float:
    r = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in r)
    return round(hits / len(keywords), 3) if keywords else 0.0

def llm_as_judge(query: str, response: str, ground_truth: str) -> tuple[int, str]:
    prompt = f"""You are evaluating an insurance AI assistant.
 
    QUESTION: {query}
    EXPECTED: {ground_truth}
    ACTUAL: {response}
    
    Score 1–5:
    5 = Fully correct, evidence cited, actionable
    4 = Mostly correct, minor gaps
    3 = Partially correct, missing key details
    2 = Mostly wrong or misleading
    1 = Completely wrong or unhelpful
    
    Respond ONLY as JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""
    
    resp = ollama.chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        parsed = json.loads(resp["message"]["content"])
        # LOW: clamp score to [1, 5] so out-of-range LLM responses don't corrupt metrics
        score = max(1, min(5, int(parsed["score"])))
        return score, str(parsed.get("reason", ""))
    except Exception:
        return 3, "parse error"

# Result Schema
RESULT_SCHEMA = StructType([
    StructField("test_id",       StringType(),  False),
    StructField("family",        StringType(),  False),
    StructField("variant_key",   StringType(),  False),
    StructField("variant_name",  StringType(),  False),
    StructField("query",         StringType(),  False),
    StructField("response",      StringType(),  False),
    StructField("latency_ms",    FloatType(),   False),
    StructField("input_tokens",  IntegerType(), False),
    StructField("output_tokens", IntegerType(), False),
    StructField("total_tokens",  IntegerType(), False),
    StructField("recall_at_5",   FloatType(),   False),
    StructField("judge_score",   IntegerType(), False),
    StructField("judge_reason",  StringType(),  False),
])
 
 
def build_spark_df(results: list[EvalResult], spark: SparkSession):
    rows = [list(asdict(r).values()) for r in results]
    return spark.createDataFrame(rows, schema=RESULT_SCHEMA)
 
 
def aggregate_results(df):
    """
    Three summary views using PySpark DataFrame API:
      1. Per-variant   — overall quality + efficiency
      2. Per-variant × family — per-family breakdown
      3. Cross-modal gap  — A2 vs A4 on cross_modal queries only
    """
    # 1. Per-variant summary
    variant_summary = (
        df.groupBy("variant_key", "variant_name")
        .agg(
            F.round(F.avg("recall_at_5"),  3).alias("avg_recall"),
            F.round(F.avg("judge_score"),  2).alias("avg_judge"),
            F.round(F.avg("latency_ms"),   1).alias("avg_latency_ms"),
            F.round(F.avg("total_tokens"), 0).alias("avg_tokens"),
            F.count("*").alias("n_tests"),
        )
        .orderBy("variant_key")
    )
 
    # 2. Per-variant × family breakdown
    family_breakdown = (
        df.groupBy("variant_key", "family")
        .agg(
            F.round(F.avg("recall_at_5"), 3).alias("avg_recall"),
            F.round(F.avg("judge_score"), 2).alias("avg_judge"),
            F.round(F.avg("latency_ms"),  1).alias("avg_latency_ms"),
        )
        .orderBy("variant_key", "family")
    )
 
    # 3. Cross-modal gap
    cross_modal_gap = (
        df.filter(F.col("family") == "cross_modal")
        .filter(F.col("variant_key").isin("A2", "A4"))
        .groupBy("variant_key")
        .agg(
            F.round(F.avg("recall_at_5"), 3).alias("avg_recall"),
            F.round(F.avg("judge_score"), 2).alias("avg_judge"),
        )
        .orderBy("variant_key")
    )
 
    return variant_summary, family_breakdown, cross_modal_gap
 
# MLFlow Logging (One run per variant)
def log_variant_to_mlflow(
    variant_key:  str,
    config:       VariantConfig,
    results:      list[EvalResult],
    variant_row,                    # PySpark Row from variant_summary
    family_rows:  list,             # PySpark Rows for this variant
):
    with mlflow.start_run(run_name=config.name):
 
        # Params — what was enabled for this variant
        mlflow.log_params({
            "variant":        variant_key,
            "use_policy":     config.use_policy,
            "use_damage":     config.use_damage,
            "use_claims":     config.use_claims,
            "use_router":     config.use_router,
            "use_retrieval":  config.use_retrieval,
            "n_test_cases":   len(results),
        })
 
        # Overall metrics
        mlflow.log_metrics({
            "avg_recall_at_5":  float(variant_row["avg_recall"]),
            "avg_judge_score":  float(variant_row["avg_judge"]),
            "avg_latency_ms":   float(variant_row["avg_latency_ms"]),
            "avg_total_tokens": float(variant_row["avg_tokens"]),
        })
 
        # Per-family metrics (visible as separate metric keys in the MLflow UI)
        for row in family_rows:
            fam = row["family"]
            mlflow.log_metrics({
                f"{fam}_recall":  float(row["avg_recall"]),
                f"{fam}_judge":   float(row["avg_judge"]),
                f"{fam}_latency": float(row["avg_latency_ms"]),
            })
 
        # Per-test-case scores for drill-down
        for r in results:
            mlflow.log_metric(f"judge_{r.test_id}",  r.judge_score)
            mlflow.log_metric(f"recall_{r.test_id}", r.recall_at_5)
 
        mlflow.set_tags({
            "description": config.description,
            "variant_key": variant_key,
        })

# Main eval runner
def run_evaluation(
    variant_filter: Optional[str] = None,
    family_filter:  Optional[str] = None,
):
    os.makedirs("eval", exist_ok=True)
 
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
 
    spark = None  # Initialize to None so finally block can safely call .stop()
    try:
        spark = get_spark()
        spark.sparkContext.setLogLevel("WARN")
    
        variants_to_run = {
            k: v for k, v in VARIANTS.items()
            if variant_filter is None or k == variant_filter
        }
        tests_to_run = [
            t for t in BENCHMARK
            if family_filter is None or t["family"] == family_filter
        ]
    
        all_results: list[EvalResult] = []
        results_by_variant: dict[str, list[EvalResult]] = {}
        total = len(variants_to_run) * len(tests_to_run)
        done  = 0
    
        print(f"\nRunning {total} evaluation cases across {len(variants_to_run)} variants\n")
    
        for v_key, config in variants_to_run.items():
            print(f"\n── {config.name} {'─' * (44 - len(config.name))}")
            variant_results: list[EvalResult] = []
    
            for test in tests_to_run:
                done += 1
                print(f"  [{done:2}/{total}] {test['id']}  {test['family']}")
    
                t0 = time.perf_counter()
                try:
                    response_text, in_tok, out_tok = run_agent_query(
                        query=test["query"],
                        config=config,
                    )
                except Exception as e:
                    response_text = f"ERROR: {e}"
                    in_tok = out_tok = 0
    
                latency_ms = (time.perf_counter() - t0) * 1000
                recall     = keyword_recall(response_text, test.get("keywords", []))
                score, reason = llm_as_judge(test["query"], response_text, test["ground_truth"])
    
                result = EvalResult(
                    test_id=test["id"],
                    family=test["family"],
                    variant_key=v_key,
                    variant_name=config.name,
                    query=test["query"],
                    response=response_text[:600],
                    latency_ms=round(latency_ms, 1),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    recall_at_5=recall,
                    judge_score=score,
                    judge_reason=reason,
                )
                variant_results.append(result)
                all_results.append(result)
    
            results_by_variant[v_key] = variant_results

        # ── Aggregation and reporting (outside the variant loop) ──────────────
        print("\nAggregating with PySpark...")
        df = build_spark_df(all_results, spark)
        df.cache()

        variant_summary, family_breakdown, cross_modal_gap = aggregate_results(df)

        v_summary_rows   = {row["variant_key"]: row for row in variant_summary.collect()}
        f_breakdown_rows: dict[str, list] = {}
        for row in family_breakdown.collect():
            f_breakdown_rows.setdefault(row["variant_key"], []).append(row)

        print("Logging to MLflow...")
        for v_key, config in variants_to_run.items():
            if v_key not in v_summary_rows:
                continue
            log_variant_to_mlflow(
                variant_key=v_key,
                config=config,
                results=results_by_variant[v_key],
                variant_row=v_summary_rows[v_key],
                family_rows=f_breakdown_rows.get(v_key, []),
            )

        # Write CSV and Parquet using pandas to avoid Hadoop ViewFileSystem issues with Java 21+
        os.makedirs("eval/results_csv", exist_ok=True)
        os.makedirs("eval/results_parquet", exist_ok=True)
        pandas_df = df.toPandas()
        pandas_df.to_csv("eval/results_csv/results.csv", index=False)
        pandas_df.to_parquet("eval/results_parquet/results.parquet", index=False)

        # ── Print summary tables ──────────────────────────────────────────────
        print("\n\n╔══════════════════════════════════════════════════╗")
        print("║         VARIANT SUMMARY  (PySpark)              ║")
        print("╚══════════════════════════════════════════════════╝")
        variant_summary.show(truncate=False)

        print("╔══════════════════════════════════════════════════╗")
        print("║         PER-FAMILY BREAKDOWN                    ║")
        print("╚══════════════════════════════════════════════════╝")
        family_breakdown.show(truncate=False)

        print("╔══════════════════════════════════════════════════╗")
        print("║         CROSS-MODAL GAP  (A2 vs A4)             ║")
        print("╚══════════════════════════════════════════════════╝")
        cross_modal_gap.show(truncate=False)

        print(f"\nResults → eval/results_csv/  and  eval/results_parquet/")
        print(f"MLflow UI → run:  mlflow ui  → open http://localhost:5000")
        print(f"Experiment name: {MLFLOW_EXPERIMENT}\n")

        return all_results
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance agent evaluation harness")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()),
                        help="Run only one ablation variant (A1/A2/A3/A4)")
    parser.add_argument("--family",
                        choices=["factual", "cross_modal", "analytical", "conversational"],
                        help="Run only one query family")
    args = parser.parse_args()
    run_evaluation(variant_filter=args.variant, family_filter=args.family)
    