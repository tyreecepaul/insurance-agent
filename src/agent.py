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
import json
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from tools import search_policy, search_damage, search_claims, RetrievalResult

load_dotenv()

# LLM
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=os.getenv("OLLAMA_BASE_URL", "https://localhost:11434"),
    num_predict=1024   
)
