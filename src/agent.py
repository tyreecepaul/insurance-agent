"""
agent.py
- LangGraph-orchestrated insurance claim agent

Pipleline:
- User Input
    - memory_node
    - router_node
    - retrieval_node
    - generator_node
    - response

python src/agent.py
"""

import os
import re
import json
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from .tools import search_policy, search_damage, search_claims, RetrievalResult

load_dotenv()

import warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")

# LLM
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    num_predict=1024   
)

# State
class AgentState(TypedDict):
    # Conversation history (LangGraph manages appending)
    messages:Annotated[list, add_messages]
 
    # Router output — one of 4 families
    query_type: str          # "factual" | "cross_modal" | "analytical" | "conversational"
 
    # Partially-filled claim draft (persists across turns)
    claim_draft: dict
 
    # Retrieved context passed to generator
    retrieved_docs: list[dict]
 
    # Optional: path to uploaded image
    image_path: Optional[str]
 
    # Detected metadata from conversation
    detected_policy_number: Optional[str]
    detected_insurance_type: Optional[str]
 
 

# Node 1 - Memory and State 

def memory_node(state: AgentState) -> AgentState:
    """
    Extract structured entities from the latest user message and update the
    claim draft. In a full production system this would use NER / slot filling.
    """
    last_message = state["messages"][-1] if state["messages"] else ""
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
 
    draft = state.get("claim_draft", {})
 
    # Simple keyword-based entity extraction (replace with NER for production)
    content_lower = content.lower()
 
    if "motor" in content_lower or "car" in content_lower or "vehicle" in content_lower:
        state["detected_insurance_type"] = "motor"
        draft["insurance_type"] = "motor"
    elif "home" in content_lower or "house" in content_lower or "property" in content_lower:
        state["detected_insurance_type"] = "home"
        draft["insurance_type"] = "home"
    elif "health" in content_lower or "medical" in content_lower or "hospital" in content_lower:
        state["detected_insurance_type"] = "health"
        draft["insurance_type"] = "health"
 
    # Extract policy numbers like POL-MOTOR-1042
    pol_match = re.search(r"POL-[A-Z]+-\d+", content, re.IGNORECASE)
    if pol_match:
        state["detected_policy_number"] = pol_match.group(0).upper()
        draft["policy_number"] = pol_match.group(0).upper()
 
    # Extract incident descriptions
    if any(w in content_lower for w in ["crash", "collision", "hit", "rear-ended", "accident"]):
        draft["incident_type"] = draft.get("incident_type", "collision")
    elif any(w in content_lower for w in ["flood", "water", "burst pipe", "leak"]):
        draft["incident_type"] = draft.get("incident_type", "water_damage")
    elif any(w in content_lower for w in ["theft", "stolen", "break-in", "burglary"]):
        draft["incident_type"] = draft.get("incident_type", "theft")
    elif any(w in content_lower for w in ["fire", "smoke", "burn"]):
        draft["incident_type"] = draft.get("incident_type", "fire")
    elif any(w in content_lower for w in ["storm", "hail", "wind", "cyclone"]):
        draft["incident_type"] = draft.get("incident_type", "storm_damage")
    elif any(w in content_lower for w in ["windscreen", "windshield", "crack", "chip"]):
        draft["incident_type"] = draft.get("incident_type", "windscreen")
 
    # Extract claimant name if "my name is X" pattern
    name_match = re.search(r"(?:my name is|i[''']m|i am)\s+([A-Z][a-z]+ [A-Z][a-z]+)", content, re.IGNORECASE)
    if name_match:
        draft["claimant_name"] = name_match.group(1)
 
    state["claim_draft"] = draft
    return state
 
 
# Node 2: Query Router
 
ROUTER_PROMPT = """You are a query classifier for an insurance claims agent.
 
Classify the user's latest message into exactly one of these 4 types:
 
1. "factual"       — User wants specific info from a policy (excess amount, what is covered, 
                     definitions, exclusions). Example: "What is my excess for windscreen damage?"
 
2. "cross_modal"   — User has uploaded or described a damage photo and wants assessment.
                     Example: "Here is a photo of my car. Is this damage covered?"
 
3. "analytical"    — Multi-step reasoning required: filing a new claim, understanding a process,
                     or complex "what should I do" questions.
                     Example: "My car was stolen last night. What do I need to do?"
 
4. "conversational"— Status enquiry on an existing claim, follow-up in a claim filing conversation,
                     or general chat. Example: "What's the status of my claim CLM-2024-001?"
 
Respond with ONLY the type word (no explanation): factual | cross_modal | analytical | conversational
"""
 
 
def router_node(state: AgentState) -> AgentState:
    """Classify the query type using a fast LLM call."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_human:
        state["query_type"] = "conversational"
        return state
 
    # If there's an image attached, force cross_modal
    if state.get("image_path"):
        state["query_type"] = "cross_modal"
        return state
 
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_human.content),
    ])
    raw = response.content.strip().lower()
 
    if "factual" in raw:
        state["query_type"] = "factual"
    elif "cross_modal" in raw:
        state["query_type"] = "cross_modal"
    elif "analytical" in raw:
        state["query_type"] = "analytical"
    else:
        state["query_type"] = "conversational"
 
    return state
 
 
# Node 3 - Retrieval Planner
def retrieval_node(state: AgentState) -> AgentState:
    """
    Select and invoke the right retrieval tools based on query type.
 
    factual      → policy search (BM25 + dense)
    cross_modal  → damage search (CLIP) + policy search
    analytical   → policy search + claims search
    conversational → claims search (status lookup)
    """
    query_type  = state["query_type"]
    last_human  = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query_text  = last_human.content if last_human else ""
    ins_type    = state.get("detected_insurance_type")
    pol_number  = state.get("detected_policy_number")
 
    retrieved: list[dict] = []
 
    def add_results(results: list[RetrievalResult]):
        for r in results:
            retrieved.append({
                "source":   r.source,
                "doc_id":   r.doc_id,
                "content":  r.content,
                "metadata": r.metadata,
                "score":    r.score,
            })
 
    if query_type == "factual":
        add_results(search_policy(query_text, insurance_type=ins_type))
 
    elif query_type == "cross_modal":
        img = state.get("image_path")
        if img:
            add_results(search_damage(image_path=img))
        else:
            add_results(search_damage(text_query=query_text))
        # Also pull relevant policy sections
        add_results(search_policy(query_text, insurance_type=ins_type, top_k=3))
 
    elif query_type == "analytical":
        add_results(search_policy(query_text, insurance_type=ins_type))
        add_results(search_claims(query_text, insurance_type=ins_type))
 
    elif query_type == "conversational":
        add_results(search_claims(
            query_text,
            policy_number=pol_number,
            insurance_type=ins_type,
        ))
 
    state["retrieved_docs"] = retrieved
    return state
 
 
# Node 4 - Generator
 
SYSTEM_PROMPT = """You are an insurance claims assistant. Your job is to help users:
1. Understand their insurance coverage
2. File new claims step by step
3. Check on existing claim status
4. Assess whether damage is likely covered
 
RULES:
- Always base your answers on the retrieved policy documents and claim records provided.
- Cite specific policy sections when stating what is or isn't covered.
- If information is not in the retrieved context, say so honestly — do NOT make up coverage decisions.
- When helping file a claim, collect information progressively: incident type → date → description → damage details → supporting documents.
- Be clear about what the user needs to do next.
- Keep responses concise and actionable.
"""
 
 
def generator_node(state: AgentState) -> AgentState:
    """
    Produce a grounded answer using retrieved context and conversation history.
    """
    retrieved = state.get("retrieved_docs", [])
    claim_draft = state.get("claim_draft", {})
    query_type  = state.get("query_type", "conversational")
 
    # Build context block from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved[:6]):  # cap at 6 chunks
        source  = doc["source"].upper()
        content = doc["content"]
        meta    = doc.get("metadata", {})
 
        if source == "POLICY":
            label = f"[Policy — {meta.get('insurer', '')} {meta.get('insurance_type', '')} p.{meta.get('page_number', '?')}]"
        elif source == "DAMAGE":
            label = f"[Damage photo — {meta.get('damage_type', '')}]"
        else:  # claims
            label = f"[Claim {meta.get('claim_id', '')} — {meta.get('claim_status', '')}]"
 
        context_parts.append(f"{label}\n{content}")
 
    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
 
    # Include claim draft if one is in progress
    draft_str = ""
    if claim_draft and any(claim_draft.values()):
        draft_str = f"\n\nCURRENT CLAIM DRAFT (collected so far):\n{json.dumps(claim_draft, indent=2)}"
 
    # Build the full prompt
    system = SystemMessage(content=SYSTEM_PROMPT)
    context_msg = HumanMessage(content=f"RETRIEVED CONTEXT:\n\n{context_str}{draft_str}")
 
    # Include recent conversation (last 6 turns)
    recent_messages = state["messages"][-6:]
 
    response = llm.invoke([system, context_msg] + recent_messages)
 
    return {"messages": [response]}
 

# Build Graph
 
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
 
    graph.add_node("memory",     memory_node)
    graph.add_node("router",     router_node)
    graph.add_node("retrieval",  retrieval_node)
    graph.add_node("generator",  generator_node)
 
    graph.set_entry_point("memory")
    graph.add_edge("memory",    "router")
    graph.add_edge("router",    "retrieval")
    graph.add_edge("retrieval", "generator")
    graph.add_edge("generator", END)
 
    return graph.compile()
 
 
# CLI
 
def run_cli():
    """Simple REPL for testing the agent in the terminal."""
    agent = build_graph()
 
    state: AgentState = {
        "messages":               [],
        "query_type":             "conversational",
        "claim_draft":            {},
        "retrieved_docs":         [],
        "image_path":             None,
        "detected_policy_number": None,
        "detected_insurance_type": None,
    }
 
    print("\nInsurance Claims Agent")
    print("Commands: 'quit' to exit | 'image <path>' to attach a photo")
    
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "q", "exit"):
                break
            if user_input.lower().startswith("image "):
                state["image_path"] = user_input[6:].strip()
                print(f"  ✓ Image attached: {state['image_path']}\n")
                continue
    
            state["messages"].append(HumanMessage(content=user_input))
            state = agent.invoke(state)
    
            last_ai = next(
                (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            if last_ai:
                print(f"\nAgent [{state['query_type']}]: {last_ai.content}\n")
    
            # Reset image after use
            state["image_path"] = None
    except KeyboardInterrupt:
        print("\n\nSession ended. Goodbye!")
 
 
if __name__ == "__main__":
    run_cli()
