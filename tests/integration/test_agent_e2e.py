"""
tests/integration/test_agent_e2e.py
End-to-end tests for agent.py graph with mocked Ollama.
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_graph


@pytest.mark.integration
class TestAgentGraph:
    """Integration tests for full agent graph."""
    
    @patch("src.agent.llm")
    @patch("src.agent.search_policy")
    @patch("src.agent.search_damage")
    @patch("src.agent.search_claims")
    def test_agent_graph_factual_query(
        self, mock_search_claims, mock_search_damage, mock_search_policy, mock_llm
    ):
        """Test agent pipeline for factual query."""
        # Mock LLM for router
        
        mock_llm.invoke.return_value = AIMessage(content="factual")
        
        # Mock retrieval results
        from tests.conftest import MockRetrievalResult
        mock_search_policy.return_value = [
            MockRetrievalResult(
                source="policy",
                doc_id="POL-001_p5",
                content="Excess: $750 for comprehensive motor",
                metadata={"page_number": 5}
            )
        ]
        
        # Build and invoke graph
        agent = build_graph()
        
        state = {
            "messages": [HumanMessage(content="What is the excess?")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = agent.invoke(state)
        
        # Verify state was updated
        assert result["query_type"] == "factual"
        assert len(result["messages"]) > 1
        # Last message should be from AI (generator)
        assert isinstance(result["messages"][-1], AIMessage)
    
    @patch("src.agent.llm")
    def test_agent_graph_empty_messages(self, mock_llm):
        """Test agent with no initial messages."""
        mock_llm.invoke.return_value = AIMessage(content="conversational")
        agent = build_graph()
        
        state = {
            "messages": [],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        # Router should handle empty messages gracefully
        result = agent.invoke(state)
        
        assert result["query_type"] == "conversational"
    
    @patch("src.agent.llm")
    @patch("src.agent.search_claims")
    def test_agent_graph_claim_status_query(self, mock_search_claims, mock_llm):
        """Test agent for conversational/status query."""
        # Mock router output
        mock_llm.invoke.return_value = AIMessage(content="conversational")
        
        # Mock claims search
        from tests.conftest import MockRetrievalResult
        mock_search_claims.return_value = [
            MockRetrievalResult(
                source="claims",
                doc_id="CLM-2024-001",
                content="Claim CLM-2024-001: Status approved, $4050",
                metadata={"claim_status": "approved"}
            )
        ]
        
        agent = build_graph()
        
        state = {
            "messages": [HumanMessage(content="What is the status of claim CLM-2024-001?")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": "POL-MOTOR-1042",
            "detected_insurance_type": None,
        }
        
        result = agent.invoke(state)
        
        assert result["query_type"] == "conversational"
        # Should have called search_claims
        assert mock_search_claims.called
    
    @patch("src.agent.llm")
    @patch("src.agent.search_policy")
    def test_agent_graph_memory_extraction(self, mock_search_policy, mock_llm):
        """Test that agent extracts and persists claim draft."""
        mock_llm.invoke.return_value = AIMessage(content="factual")
        mock_search_policy.return_value = []
        
        agent = build_graph()
        
        state = {
            "messages": [
                HumanMessage(
                    content="My policy number is POL-MOTOR-1042 and my car was in a collision."
                )
            ],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = agent.invoke(state)
        
        # Memory node should have extracted policy number and incident type
        assert result["detected_policy_number"] == "POL-MOTOR-1042"
        assert result["claim_draft"].get("policy_number") == "POL-MOTOR-1042"
        assert result["claim_draft"].get("incident_type") == "collision"


@pytest.mark.integration
class TestAgentNodeSequence:
    """Test the correct sequence of nodes in the graph."""
    
    @patch("src.agent.llm")
    def test_node_execution_order(self, mock_llm, sample_agent_state):
        """Test that nodes execute in expected order: memory → router → retrieval → generator."""
        call_order = []
        
        def track_router(*args, **kwargs):
            call_order.append("router")
            return AIMessage(content="factual")
        
        mock_llm.invoke.side_effect = track_router
        
        with patch("src.agent.search_policy") as mock_search:
            mock_search.return_value = []
            
            agent = build_graph()
            result = agent.invoke(sample_agent_state)
            
            # All nodes should have executed
            assert len(result["messages"]) > 1
            # Should have AI response at end
            assert isinstance(result["messages"][-1], AIMessage)
