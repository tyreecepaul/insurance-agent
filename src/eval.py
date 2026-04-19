"""
eval.py
- Evaluation harness with pandas aggregation and MLflow experiment tracking

- Tests query families across ablation variants (A1-A4)
- MLflow tracks parameters, metrics and artefacts every run
- pandas aggregates results into summary tables

Usage:
    python src/eval.py                           # full ablation study
    python src/eval.py --variant A4              # single variant
    python src/eval.py --variants A1 A4          # multiple specific variants
    python src/eval.py --family factual          # single query family
    python src/eval.py --dry-run                 # validate benchmark, no LLM calls
    python src/eval.py --summary-only            # reprint last run's tables
    mlflow ui                                    # view results at localhost:5000
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from typing import Optional
from dataclasses import dataclass, asdict

import mlflow
import pandas as pd
import ollama
from dotenv import load_dotenv

from src.tools import search_policy, search_damage, search_claims
from src.agent import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT

load_dotenv()

MLFLOW_EXPERIMENT = "insurance-agent-ablation"

# ── Benchmark ─────────────────────────────────────────────────────────────────
#
# Fields:
#   id             — unique case identifier
#   family         — factual | cross_modal | analytical | conversational | multi_turn
#   query          — the user question (final turn for multi_turn cases)
#   ground_truth   — expected answer text used by LLM judge
#   expected_source — policy | damage | claims
#   keywords       — list of strings for recall@5 scoring
#   image_path     — (cross_modal only) path to a real image fixture
#   prior_turns    — (multi_turn only) list of {"role": str, "content": str} dicts
#                    representing conversation history before the final query

BENCHMARK: list[dict] = [
    # ── FAMILY 1: Factual retrieval ──────────────────────────────────────────
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

    # ── FAMILY 2: Cross-modal assessment ─────────────────────────────────────
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
    # F2-03 and F2-04 use real indexed motor damage photos so CLIP produces
    # meaningful embeddings. Blank placeholder images produce near-random
    # cosine similarities and make cross-modal results uninterpretable.
    {
        "id": "F2-03",
        "family": "cross_modal",
        "query": "Here is a photo showing crash damage to my vehicle on the highway. Is this covered under my policy?",
        "image_path": "data/damage_photos/10.10.09_motorcade_governor_crash_highway_001.jpg",
        "ground_truth": "Vehicle crash damage on a highway is covered under comprehensive motor insurance, subject to the standard excess.",
        "expected_source": "damage",
        "keywords": ["crash", "covered", "comprehensive", "motor", "excess"],
    },
    {
        "id": "F2-04",
        "family": "cross_modal",
        "query": "Can you assess this vehicle damage from a collision? What would my claim cover?",
        "image_path": "data/damage_photos/640px-Wrecked_car_at_Tuntorp,_Brastad_1.jpg",
        "ground_truth": "Collision damage to a vehicle is covered under comprehensive motor insurance. The claim covers repair costs minus the applicable excess.",
        "expected_source": "damage",
        "keywords": ["collision", "covered", "comprehensive", "motor", "repair"],
    },

    # ── FAMILY 3: Analytical / multi-hop ─────────────────────────────────────
    # Recall is a secondary signal on analytical queries. LLMs paraphrase
    # procedures rather than repeating indexed text verbatim, so keyword recall
    # systematically underestimates answer quality. The primary quality signal
    # for this family is judge_score.
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
    {
        "id": "F3-03",
        "family": "analytical",
        "query": "What steps do I need to follow to claim for a hospital stay?",
        "ground_truth": "Steps: 1. Obtain pre-approval from health fund for elective procedures 2. Confirm hospital is in fund network 3. Submit itemised invoice with claim form 4. Include Medicare statement 5. Keep all receipts.",
        "expected_source": "policy",
        # "itemised"/"itemized" spelling varies by LLM locale — use "itemi" (substring of both).
        # "itemis" only matches British spelling; "itemi" matches "itemised" and "itemized".
        # "approval" matches "pre-approval", "prior approval", and "preapproval".
        "keywords": ["approval", "hospital", "fund", "claim form", "itemi", "Medicare"],
    },
    {
        "id": "F3-04",
        "family": "analytical",
        "query": "My laptop was stolen from my car — what do I do now?",
        "ground_truth": "Steps: 1. File a police report immediately and obtain report number 2. Record serial numbers of stolen items 3. Gather purchase receipts 4. Contact insurer and provide claim number from police 5. Submit itemised claim.",
        "expected_source": "policy",
        # "contact insurer" fails when LLM writes "contact your insurer" (substring mismatch).
        # Split into "insurer" alone — "police" already covers the contact action.
        "keywords": ["police", "report", "serial", "receipt", "insurer", "claim"],
    },
    {
        "id": "F3-05",
        "family": "analytical",
        "query": "How do I get my windscreen repaired under my policy?",
        "ground_truth": "Contact insurer to lodge a windscreen claim. You will be directed to an approved repairer. No excess applies for windscreen repair under comprehensive cover.",
        "expected_source": "policy",
        "keywords": ["windscreen", "approved repairer", "excess", "contact", "repair"],
    },

    # ── FAMILY 4: Conversational / status ────────────────────────────────────
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

    # ── FAMILY 5: Multi-turn — tests memory node entity extraction ────────────
    #
    # prior_turns:  conversation history before the evaluated query.
    #               These are injected into the LLM messages list so the model
    #               has context. The eval scores only the final query's response.
    #
    # Note: the memory_node guard (messages <= 1) means single-turn eval queries
    # bypass entity extraction. Multi-turn cases here exercise the full path
    # when prior_turns are injected as LLM context. A future eval iteration
    # should wire prior turns through the full LangGraph state to properly
    # test slot-filling persistence across turns.
    {
        "id": "MT-01",
        "family": "multi_turn",
        "prior_turns": [
            {
                "role": "user",
                "content": (
                    "Hi, my name is Jordan Lee and my policy number is POL-MOTOR-1042. "
                    "My car was rear-ended on Pacific Motorway this morning."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I'm sorry to hear that, Jordan. I've noted your motor policy "
                    "POL-MOTOR-1042 and the collision on Pacific Motorway. To begin "
                    "your claim, could you tell me whether the other driver stopped "
                    "and exchanged details?"
                ),
            },
        ],
        "query": "Yes they did. What information do I still need to provide to complete my claim?",
        "ground_truth": (
            "For motor collision claim under POL-MOTOR-1042, still required: "
            "incident date and time, third party driver details (name, licence, "
            "insurance), photos of the damage, and a repair quote from an approved repairer."
        ),
        "expected_source": "claims",
        "keywords": ["POL-MOTOR-1042", "motor", "third party", "damage", "repair"],
    },
    {
        "id": "MT-02",
        "family": "multi_turn",
        "prior_turns": [
            {
                "role": "user",
                "content": (
                    "I have a home insurance policy POL-HOME-2871. A storm last week "
                    "cracked several roof tiles and water is getting in."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I've noted your home policy POL-HOME-2871 and the storm damage. "
                    "Your first priority is to prevent further damage — have you been "
                    "able to arrange temporary tarping of the affected area?"
                ),
            },
        ],
        "query": "Yes, I've tarped the roof and kept all the receipts. What are my next steps and will the tarping cost be reimbursed?",
        "ground_truth": (
            "Next steps for POL-HOME-2871 storm claim: lodge a formal claim with your "
            "insurer, submit tarping receipts as emergency expenses (covered under your "
            "home policy), arrange an assessor inspection, and photograph all visible "
            "damage before making further repairs."
        ),
        "expected_source": "claims",
        # "receipts" fails when LLM uses singular "receipt" — use singular as it
        # is a substring of both "receipt" and "receipts".
        "keywords": ["POL-HOME-2871", "tarp", "emergency", "receipt", "covered", "assess"],
    },

    # ── Additional Factual Cases (Policy Details) ──────────────────────────
    {
        "id": "F1-04",
        "family": "factual",
        "query": "Are rental cars covered under my motor insurance policy while mine is being repaired?",
        "ground_truth": "Rental car cover is available as an optional extra. If included, it covers hire car costs up to a daily limit while your vehicle is being repaired after a covered claim.",
        "expected_source": "policy",
        "keywords": ["rental", "hire car", "covered", "daily limit", "repair"],
    },
    {
        "id": "F1-05",
        "family": "factual",
        "query": "What is the maximum benefit for loss of income in my income protection insurance?",
        "ground_truth": "Income protection replaces up to 75% of your gross weekly income, subject to the waiting period (usually 14 or 30 days) and benefit period (2 years or to age 65, whichever is earlier).",
        "expected_source": "policy",
        "keywords": ["income", "75%", "weekly", "waiting period", "benefit"],
    },
    {
        "id": "F1-06",
        "family": "factual",
        "query": "Is repatriation from overseas covered if I fall ill while traveling?",
        "ground_truth": "Travel insurance includes repatriation cover for emergency medical evacuation and return to your home country. This covers air ambulance, medical escort, and commercial flights home if hospitalised.",
        "expected_source": "policy",
        "keywords": ["repatriation", "overseas", "evacuation", "ambulance", "hospitali"],
    },
    {
        "id": "F1-07",
        "family": "factual",
        "query": "What is the deductible for contents insurance if I make a claim?",
        "ground_truth": "Standard contents deductible is $250 per claim. A higher voluntary deductible (e.g. $500 or $1000) may reduce your premium but increases your out-of-pocket cost when you claim.",
        "expected_source": "policy",
        "keywords": ["deductible", "250", "voluntary", "claim", "premium"],
    },
    {
        "id": "F1-08",
        "family": "factual",
        "query": "Is theft of items from my car covered under home contents insurance?",
        "ground_truth": "Items stolen from a locked vehicle may be covered under your home contents or personal property cover, but your car insurance excess applies. Items should be insured for their full replacement value.",
        "expected_source": "policy",
        "keywords": ["theft", "car", "vehicle", "contents", "locked", "excess"],
    },

    # ── Additional Cross-Modal Cases (Damage Assessment) ───────────────────
    {
        "id": "F2-05",
        "family": "cross_modal",
        "query": "I have a photo of a large dent in my car door after it was struck in a car park. Is this covered by my insurance?",
        "ground_truth": "Door dent damage is covered under comprehensive motor insurance. Repair costs are covered minus the excess, and paintwork damage from the impact is also included.",
        "expected_source": "damage",
        "keywords": ["dent", "door", "comprehensive", "paintwork", "repair", "excess"],
    },
    {
        "id": "F2-06",
        "family": "cross_modal",
        "query": "Here's a photo of my fence that was blown over in last night's storm. Will home insurance pay to rebuild it?",
        "ground_truth": "Damage to fencing from storms is typically covered under home building insurance. The cost to rebuild is covered minus the excess, depending on your specific policy wording.",
        "expected_source": "damage",
        "keywords": ["fence", "storm", "rebuild", "building", "covered", "excess"],
    },
    {
        "id": "F2-07",
        "family": "cross_modal",
        "query": "I'm attaching a photo of my laptop that was damaged when I spilled coffee on it. Is accidental damage covered?",
        "ground_truth": "Accidental damage coverage is optional and must be added to your home contents policy. With this cover, spill damage to electronics like laptops is covered minus the excess.",
        "expected_source": "damage",
        "keywords": ["accidental damage", "laptop", "spill", "covered", "optional", "excess"],
    },
    {
        "id": "F2-08",
        "family": "cross_modal",
        "query": "This photo shows my home office equipment including computer, printer and desk. If there's a house fire, will all of this be covered?",
        "ground_truth": "Office equipment is covered under home contents insurance, including computer, printer, furniture and other business items, subject to any work-use exclusions or limits specified in your policy.",
        "expected_source": "damage",
        "keywords": ["office", "equipment", "fire", "contents", "computer", "covered"],
    },

    # ── Additional Analytical Cases (Procedures) ──────────────────────────
    {
        "id": "F3-06",
        "family": "analytical",
        "query": "My car was parked and hit by another vehicle that didn't stop. How should I proceed?",
        "ground_truth": "Steps: 1. Take photos of damage and note time 2. Find witness contact details if possible 3. Report to police (hit-and-run) 4. Contact your insurer within 24h 5. Provide claim reference from police 6. Obtain repair quote.",
        "expected_source": "policy",
        "keywords": ["hit-and-run", "photo", "witness", "police", "claim", "repair"],
    },
    {
        "id": "F3-07",
        "family": "analytical",
        "query": "What should I do immediately after a motor vehicle accident to protect my claim?",
        "ground_truth": "Immediate steps: 1. Ensure safety (move to safe location if possible) 2. Call police if injuries 3. Exchange details with other drivers 4. Take photos of all damage 5. Get contact details of witnesses 6. Note exact location and time 7. Don't admit liability.",
        "expected_source": "policy",
        "keywords": ["accident", "safety", "police", "exchange", "photo", "witnesses", "liability"],
    },
    {
        "id": "F3-08",
        "family": "analytical",
        "query": "How do I claim for medical expenses I incurred due to an accident?",
        "ground_truth": "Steps: 1. Obtain itemised invoices from all healthcare providers 2. Collect receipts for medications and equipment 3. Request medical reports from treating doctors 4. Submit all documents with claim form 5. Include any gap insurance receipts 6. Allow 4-6 weeks for assessment.",
        "expected_source": "policy",
        "keywords": ["medical", "invoice", "receipt", "doctor", "report", "claim form", "weeks"],
    },
    {
        "id": "F3-09",
        "family": "analytical",
        "query": "What is the process for disputing a claim rejection by my insurer?",
        "ground_truth": "Dispute process: 1. Request written explanation of rejection 2. Review policy wording and assessment 3. Gather supporting evidence 4. Lodge written complaint with insurer 5. If unresolved, refer to Insurance Ombudsman 6. Provide all documents and correspondence.",
        "expected_source": "policy",
        "keywords": ["dispute", "rejection", "complaint", "ombudsman", "evidence", "written"],
    },

    # ── Additional Conversational Cases (Claim Status & Interaction) ──────
    {
        "id": "F4-05",
        "family": "conversational",
        "query": "I'm Alex Park, and I need to check on my motor insurance claim that I submitted two weeks ago.",
        "ground_truth": "To locate your claim, I need your policy number and the claim reference provided when you lodged it. Typical processing time is 4-8 weeks depending on complexity. Once I have these details, I can advise on your claim status.",
        "expected_source": "claims",
        "keywords": ["policy", "claim reference", "processing", "status", "weeks"],
    },
    {
        "id": "F4-06",
        "family": "conversational",
        "query": "How long does it usually take to settle a home contents insurance claim?",
        "ground_truth": "Simple contents claims typically settle in 3-5 business days. More complex claims with assessments may take 2-4 weeks. Emergency advance payments can be arranged for urgent needs while claim is being assessed.",
        "expected_source": "claims",
        "keywords": ["settle", "contents", "days", "weeks", "assess", "advance"],
    },
    {
        "id": "F4-07",
        "family": "conversational",
        "query": "My claim was approved. What happens next and when will I get paid?",
        "ground_truth": "After approval: 1. Settlement amount is confirmed 2. Payment is processed to your nominated bank account 3. Direct bank transfer typically takes 2-3 business days 4. You'll receive settlement letter and receipt via post or email.",
        "expected_source": "claims",
        "keywords": ["approved", "settlement", "payment", "bank", "transfer", "business days"],
    },
    {
        "id": "F4-08",
        "family": "conversational",
        "query": "Can I appoint a lawyer or representative to handle my claim?",
        "ground_truth": "Yes. You can appoint a lawyer, loss adjuster, or claims agent to represent you. Provide a signed letter of authority to your representative and forward it to the insurer. Their fees may be claimed back as part of your settlement.",
        "expected_source": "claims",
        "keywords": ["lawyer", "representative", "authority", "adjuster", "settlement"],
    },
    {
        "id": "F4-09",
        "family": "conversational",
        "query": "I disagree with the damage assessment on my car repair claim. What can I do?",
        "ground_truth": "If you disagree with the repair quote: 1. Obtain independent assessment from another repairer 2. Submit the alternative quote to your insurer 3. Insurer will review both quotes 4. If values differ significantly, an umpire may be appointed. You can also lodge a formal dispute.",
        "expected_source": "claims",
        "keywords": ["assessment", "quote", "independent", "umpire", "dispute", "repairer"],
    },

    # ── Additional Multi-Turn Cases (Conversation Continuations) ───────────
    {
        "id": "MT-03",
        "family": "multi_turn",
        "prior_turns": [
            {
                "role": "user",
                "content": (
                    "Hi, I've had a car accident. I have motor policy POL-MOTOR-5543. "
                    "Another driver was fully at fault."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I'm sorry to hear about your accident. I've noted your motor policy "
                    "POL-MOTOR-5543 and that the other driver was at fault. I'll help you "
                    "lodge a claim. Do you have the other driver's insurance details?"
                ),
            },
        ],
        "query": "Yes, I have their insurer details. Can I claim directly from their insurance instead of mine?",
        "ground_truth": (
            "Yes, you can lodge a third-party claim directly with the at-fault driver's "
            "insurer. However, claiming through your own insurer (POL-MOTOR-5543) may be "
            "faster as they handle all communication. Your insurer will recover the cost "
            "from the third party, so you won't pay your excess."
        ),
        "expected_source": "claims",
        "keywords": ["third-party", "insurer", "claim", "excess", "recover", "at-fault"],
    },
    {
        "id": "MT-04",
        "family": "multi_turn",
        "prior_turns": [
            {
                "role": "user",
                "content": (
                    "I have health insurance with HealthCare Plus. I need a knee surgery "
                    "that my doctor recommended. My policy is POL-HEALTH-7821."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I've noted your health policy POL-HEALTH-7821. Before you book the "
                    "surgery, I recommend checking hospital accreditation and obtaining "
                    "pre-approval. Have you chosen a hospital yet?"
                ),
            },
        ],
        "query": "The doctor recommended the hospital, but it's quite expensive. Can I choose a different hospital to save money?",
        "ground_truth": (
            "Yes, under your POL-HEALTH-7821 you can choose any hospital in the fund's "
            "network to potentially reduce costs. However, verify that: 1) The surgeon is "
            "on your fund's list 2) The new hospital is accredited 3) Anaesthetist and "
            "hospital theatre fees are covered. Compare quotes between network hospitals."
        ),
        "expected_source": "claims",
        "keywords": ["hospital", "network", "surgeon", "fund", "accredited", "quote", "costs"],
    },
]


# ── Ablation Variant Configs ───────────────────────────────────────────────────

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


# ── Result Dataclass ───────────────────────────────────────────────────────────

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


# ── Agent query runner ─────────────────────────────────────────────────────────

def run_agent_query(
    query: str,
    config: VariantConfig,
    image_path: Optional[str] = None,
    prior_turns: Optional[list[dict]] = None,
) -> tuple[str, int, int]:
    """
    Run a single query through a variant configuration and return
    (response_text, input_tokens, output_tokens).

    prior_turns: list of {"role": "user"|"assistant", "content": str} dicts
                 representing conversation history to prepend before the query.
                 Used by multi_turn benchmark cases.
    """
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
                # Discard low-confidence CLIP text→caption matches.
                # Filename-derived captions score < 0.30 and dilute policy context.
                if r.score >= 0.30:
                    context_parts.append(f"[DAMAGE — {r.metadata.get('damage_type','')}] {r.content[:300]}")

        if config.use_claims and family in ("conversational", "analytical", "all"):
            for r in search_claims(query, top_k=5):
                context_parts.append(f"[CLAIM {r.metadata.get('claim_id','')}] {r.content[:300]}")

    system = AGENT_SYSTEM_PROMPT
    context_str = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"

    # Build messages: system → prior conversation turns → current question
    messages: list[dict] = [{"role": "system", "content": system}]
    if prior_turns:
        # Prepend history so the LLM has context for follow-up questions.
        # This exercises LLM-level context retention; full agent state-based
        # memory (slot-filling persistence) requires running prior turns through
        # the LangGraph graph, which is deferred to a future eval cycle.
        messages.extend(prior_turns)
    messages.append({"role": "user", "content": user_msg})

    try:
        response = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            messages=messages,
        )
        text    = response["message"]["content"]
        in_tok  = response.get("prompt_eval_count", 0)
        out_tok = response.get("eval_count", 0)
        return text, in_tok, out_tok
    except Exception as e:
        return f"ERROR: {str(e)}", 0, 0


# ── Metrics ────────────────────────────────────────────────────────────────────

def keyword_recall(response: str, keywords: list[str]) -> float:
    # Normalize hyphens before matching so "third-party" matches the keyword
    # "third party" and vice versa. LLMs freely alternate between hyphenated
    # and spaced forms (e.g. "pre-approval" / "pre approval") — failing on
    # punctuation only inflates false negatives without measuring retrieval quality.
    r = response.lower().replace("-", " ")
    hits = sum(1 for kw in keywords if kw.lower().replace("-", " ") in r)
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
        score  = max(1, min(5, int(parsed["score"])))
        return score, str(parsed.get("reason", ""))
    except Exception:
        return 3, "parse error"


# ── Aggregation ────────────────────────────────────────────────────────────────
#
# pandas for local aggregation — production equivalent would use PySpark or
# BigQuery SQL over a partitioned results table at scale. Schema mirrors a
# Spark StructType: variant_key STRING, family STRING, recall_at_5 FLOAT,
# judge_score INT, latency_ms FLOAT, total_tokens INT.

def aggregate_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three summary views:
      1. Per-variant   — overall quality + efficiency
      2. Per-variant × family — per-family breakdown
      3. Cross-modal gap  — A2 vs A4 on cross_modal queries only
    """
    # 1. Per-variant summary
    variant_summary = (
        df.groupby(["variant_key", "variant_name"])
        .agg(
            avg_recall   =("recall_at_5",  lambda x: round(x.mean(), 3)),
            avg_judge    =("judge_score",   lambda x: round(x.mean(), 2)),
            avg_latency_ms=("latency_ms",   lambda x: round(x.mean(), 1)),
            avg_tokens   =("total_tokens",  lambda x: round(x.mean(), 0)),
            n_tests      =("test_id",       "count"),
        )
        .reset_index()
        .sort_values("variant_key")
        .reset_index(drop=True)
    )

    # 2. Per-variant × family breakdown
    family_breakdown = (
        df.groupby(["variant_key", "family"])
        .agg(
            avg_recall    =("recall_at_5", lambda x: round(x.mean(), 3)),
            avg_judge     =("judge_score",  lambda x: round(x.mean(), 2)),
            avg_latency_ms=("latency_ms",   lambda x: round(x.mean(), 1)),
        )
        .reset_index()
        .sort_values(["variant_key", "family"])
        .reset_index(drop=True)
    )

    # 3. Cross-modal gap
    cross_modal_gap = (
        df[
            (df["family"] == "cross_modal") &
            (df["variant_key"].isin(["A2", "A4"]))
        ]
        .groupby("variant_key")
        .agg(
            avg_recall=("recall_at_5", lambda x: round(x.mean(), 3)),
            avg_judge =("judge_score",  lambda x: round(x.mean(), 2)),
        )
        .reset_index()
        .sort_values("variant_key")
        .reset_index(drop=True)
    )

    return variant_summary, family_breakdown, cross_modal_gap


def build_spark_df(results: list[EvalResult], spark_session: "SparkSession") -> pd.DataFrame:
    """
    Convert a list of EvalResult objects to a pandas DataFrame.
    
    Note: Despite the function name 'build_spark_df', it directly creates a pandas
    DataFrame to avoid unnecessary Spark round-tripping. Spark conversion can be
    done at call sites if needed using spark_session.createDataFrame(df).
    
    Args:
        results: List of EvalResult dataclass instances
        spark_session: Active Spark session (optional for future use)
    
    Returns:
        pandas DataFrame with columns matching EvalResult fields
    """
    return pd.DataFrame([asdict(result) for result in results])


def _print_table(title: str, df: pd.DataFrame) -> None:
    """Print a DataFrame with a box-drawing banner."""
    print(f"\n╔{'═' * 50}╗")
    print(f"║  {title:<48}║")
    print(f"╚{'═' * 50}╝")
    print(df.to_string(index=False))
    print()


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_variant_to_mlflow(
    variant_key:  str,
    config:       VariantConfig,
    results:      list[EvalResult],
    variant_row:  dict,
    family_rows:  list[dict],
) -> None:
    with mlflow.start_run(run_name=config.name):
        mlflow.log_params({
            "variant":        variant_key,
            "use_policy":     config.use_policy,
            "use_damage":     config.use_damage,
            "use_claims":     config.use_claims,
            "use_router":     config.use_router,
            "use_retrieval":  config.use_retrieval,
            "n_test_cases":   len(results),
        })

        mlflow.log_metrics({
            "avg_recall_at_5":  float(variant_row["avg_recall"]),
            "avg_judge_score":  float(variant_row["avg_judge"]),
            "avg_latency_ms":   float(variant_row["avg_latency_ms"]),
            "avg_total_tokens": float(variant_row["avg_tokens"]),
        })

        for row in family_rows:
            fam = row["family"]
            mlflow.log_metrics({
                f"{fam}_recall":  float(row["avg_recall"]),
                f"{fam}_judge":   float(row["avg_judge"]),
                f"{fam}_latency": float(row["avg_latency_ms"]),
            })

        for r in results:
            mlflow.log_metric(f"judge_{r.test_id}",  r.judge_score)
            mlflow.log_metric(f"recall_{r.test_id}", r.recall_at_5)

        mlflow.set_tags({
            "description": config.description,
            "variant_key": variant_key,
        })


# ── Main eval runner ───────────────────────────────────────────────────────────

def run_evaluation(
    variant_filter:  Optional[str]       = None,
    variants_filter: Optional[list[str]] = None,
    family_filter:   Optional[str]       = None,
) -> list[EvalResult]:
    os.makedirs("eval", exist_ok=True)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    variants_to_run = VARIANTS
    if variant_filter is not None:
        variants_to_run = {k: v for k, v in VARIANTS.items() if k == variant_filter}
    elif variants_filter is not None:
        variants_to_run = {k: v for k, v in VARIANTS.items() if k in variants_filter}

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
                    image_path=test.get("image_path"),
                    prior_turns=test.get("prior_turns"),
                )
            except Exception as e:
                response_text = f"ERROR: {e}"
                in_tok = out_tok = 0

            latency_ms    = (time.perf_counter() - t0) * 1000
            recall        = keyword_recall(response_text, test.get("keywords", []))
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

    # ── Aggregation ─────────────────────────────────────────────────────────
    print("\nAggregating results...")
    df = pd.DataFrame([asdict(r) for r in all_results])

    variant_summary, family_breakdown, cross_modal_gap = aggregate_results(df)

    v_summary_rows  = {row["variant_key"]: row for row in variant_summary.to_dict("records")}
    f_breakdown_rows: dict[str, list[dict]] = {}
    for row in family_breakdown.to_dict("records"):
        f_breakdown_rows.setdefault(row["variant_key"], []).append(row)

    # ── MLflow logging ───────────────────────────────────────────────────────
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

    # ── Persist results ──────────────────────────────────────────────────────
    df.to_csv("eval/results.csv", index=False)
    df.to_parquet("eval/results.parquet", index=False)

    # ── Print summary tables ─────────────────────────────────────────────────
    _print_table("VARIANT SUMMARY", variant_summary)
    _print_table("PER-FAMILY BREAKDOWN", family_breakdown)
    _print_table("CROSS-MODAL GAP  (A2 vs A4)", cross_modal_gap)

    print(f"Results saved → eval/results.csv  and  eval/results.parquet")
    print(f"MLflow UI     → run:  mlflow ui  → open http://localhost:5000")
    print(f"Experiment    → {MLFLOW_EXPERIMENT}\n")

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────────

def _create_fixtures() -> None:
    """Create minimal PNG fixtures for cross-modal benchmark cases F2-03 and F2-04."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow required: pip install Pillow")
        sys.exit(1)

    fixtures_dir = Path("tests/fixtures/images")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Distinct colours so CLIP embeddings differ between the two images
    Image.new("RGB", (224, 224), color=(120, 140, 160)).save(fixtures_dir / "hail_damage.png")
    Image.new("RGB", (224, 224), color=(50,  100, 150)).save(fixtures_dir / "water_damage.png")
    print(f"Fixtures created in {fixtures_dir}/")


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Insurance agent evaluation harness")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Run only one ablation variant (A1/A2/A3/A4)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        help="Run specific variants (e.g. --variants A1 A4)",
    )
    parser.add_argument(
        "--family",
        choices=["factual", "cross_modal", "analytical", "conversational", "multi_turn"],
        help="Run only one query family",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all benchmark cases and print a summary — no LLM calls",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Reprint summary tables from the last saved eval/results.csv without re-running",
    )
    parser.add_argument(
        "--create-fixtures",
        action="store_true",
        help="Create image fixtures for cross-modal benchmark cases and exit",
    )
    args = parser.parse_args()

    if args.create_fixtures:
        _create_fixtures()
        sys.exit(0)

    if args.summary_only:
        csv_path = "eval/results.csv"
        if not os.path.exists(csv_path):
            print(f"No results found at {csv_path}. Run eval first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        variant_summary, family_breakdown, cross_modal_gap = aggregate_results(df)
        _print_table("VARIANT SUMMARY", variant_summary)
        _print_table("PER-FAMILY BREAKDOWN", family_breakdown)
        _print_table("CROSS-MODAL GAP  (A2 vs A4)", cross_modal_gap)
        sys.exit(0)

    if args.dry_run:
        variants_to_run = VARIANTS
        if args.variant:
            variants_to_run = {args.variant: VARIANTS[args.variant]}
        elif args.variants:
            variants_to_run = {k: VARIANTS[k] for k in args.variants}

        tests_to_run = [
            t for t in BENCHMARK
            if args.family is None or t["family"] == args.family
        ]
        total = len(variants_to_run) * len(tests_to_run)
        print(f"\nDry run: {len(tests_to_run)} cases × {len(variants_to_run)} variants = {total} total\n")
        families: dict[str, int] = {}
        for t in tests_to_run:
            families[t["family"]] = families.get(t["family"], 0) + 1
            img   = f"  [image: {t['image_path']}]" if t.get("image_path") else ""
            turns = f"  [+{len(t['prior_turns'])} prior turns]" if t.get("prior_turns") else ""
            print(f"  {t['id']:6}  {t['family']:14}  {t['query'][:55]}...{img}{turns}")
        print(f"\nCases per family: {families}")
        sys.exit(0)

    run_evaluation(
        variant_filter=args.variant,
        variants_filter=args.variants,
        family_filter=args.family,
    )
