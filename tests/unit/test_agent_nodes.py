"""
tests/unit/test_agent_nodes.py
Unit tests for agent.py nodes (memory, router, etc.)
Fast, isolated tests with mocked dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import memory_node, router_node
from src.eval import VARIANTS


@pytest.mark.unit
class TestMemoryNode:
    """Tests for memory_node entity extraction."""
    
    def test_detect_motor_insurance(self):
        """Test detection of motor insurance type."""
        state = {
            "messages": [HumanMessage(content="My car was damaged in an accident.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["detected_insurance_type"] == "motor"
        assert result["claim_draft"]["insurance_type"] == "motor"
    
    def test_detect_home_insurance(self):
        """Test detection of home insurance type."""
        state = {
            "messages": [HumanMessage(content="Water damage to my house from a burst pipe.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["detected_insurance_type"] == "home"
        assert result["claim_draft"]["insurance_type"] == "home"
    
    def test_detect_health_insurance(self):
        """Test detection of health insurance type."""
        state = {
            "messages": [HumanMessage(content="I need to claim for medical expenses.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["detected_insurance_type"] == "health"
        assert result["claim_draft"]["insurance_type"] == "health"
    
    def test_extract_policy_number(self):
        """Test extraction of policy number from message."""
        state = {
            "messages": [
                HumanMessage(content="My policy number is POL-MOTOR-1042, and my car was damaged.")
            ],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["detected_policy_number"] == "POL-MOTOR-1042"
        assert result["claim_draft"]["policy_number"] == "POL-MOTOR-1042"
    
    def test_extract_incident_type_crash(self):
        """Test incident type detection for collision."""
        state = {
            "messages": [HumanMessage(content="I was in a rear-end collision at traffic lights.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("incident_type") == "collision"
    
    def test_extract_incident_type_water_damage(self):
        """Test incident type detection for water damage."""
        state = {
            "messages": [HumanMessage(content="A burst pipe in my bathroom caused water damage.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("incident_type") == "water_damage"
    
    def test_extract_incident_type_theft(self):
        """Test incident type detection for theft."""
        state = {
            "messages": [HumanMessage(content="My car was stolen from the driveway.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("incident_type") == "theft"
    
    def test_extract_incident_type_fire(self):
        """Test incident type detection for fire."""
        state = {
            "messages": [HumanMessage(content="Kitchen fire caused smoke damage to the house.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("incident_type") == "fire"
    
    def test_extract_incident_type_storm(self):
        """Test incident type detection for storm damage."""
        state = {
            "messages": [HumanMessage(content="A hail storm damaged the roof of my house.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("incident_type") == "storm_damage"
    
    def test_extract_claimant_name(self):
        """Test extraction of claimant name."""
        state = {
            "messages": [HumanMessage(content="My name is Sarah Chen and I need to lodge a claim.")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        assert result["claim_draft"].get("claimant_name") == "Sarah Chen"
    
    def test_preserve_existing_claim_draft(self):
        """Test that existing claim draft is preserved."""
        existing_draft = {
            "policy_number": "POL-MOTOR-5000",
            "insurance_type": "motor",
        }
        state = {
            "messages": [HumanMessage(content="My car was in a crash.")],
            "query_type": "conversational",
            "claim_draft": existing_draft,
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = memory_node(state)
        
        # Original values preserved
        assert result["claim_draft"]["policy_number"] == "POL-MOTOR-5000"
        # New values added
        assert result["claim_draft"]["incident_type"] == "collision"


@pytest.mark.unit
class TestRouterNode:
    """Tests for router_node query classification."""
    
    @patch("src.agent.llm")
    def test_router_factual_query(self, mock_llm):
        """Test classification of factual query."""
        mock_llm.invoke.return_value = MagicMock(content="factual")
        
        state = {
            "messages": [HumanMessage(content="What is the excess for comprehensive motor?")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = router_node(state)
        
        assert result["query_type"] == "factual"
    
    @patch("src.agent.llm")
    def test_router_cross_modal_with_image(self, mock_llm):
        """Test that image path forces cross_modal classification."""
        state = {
            "messages": [HumanMessage(content="Is this damage covered?")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": "/path/to/damage_photo.jpg",  # Image attached
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = router_node(state)
        
        # Should be cross_modal regardless of LLM response
        assert result["query_type"] == "cross_modal"
        # LLM should not be called when image present
        mock_llm.invoke.assert_not_called()
    
    @patch("src.agent.llm")
    def test_router_analytical_query(self, mock_llm):
        """Test classification of analytical query."""
        mock_llm.invoke.return_value = MagicMock(content="analytical")
        
        state = {
            "messages": [HumanMessage(content="My car was stolen. What steps do I need to take?")],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = router_node(state)
        
        assert result["query_type"] == "analytical"
    
    @patch("src.agent.llm")
    def test_router_conversational_query(self, mock_llm):
        """Test classification of conversational query."""
        mock_llm.invoke.return_value = MagicMock(content="conversational")
        
        state = {
            "messages": [HumanMessage(content="What is the status of my claim?")],
            "query_type": "factual",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = router_node(state)
        
        assert result["query_type"] == "conversational"
    
    @patch("src.agent.llm")
    def test_router_no_messages(self, mock_llm):
        """Test router when no messages exist."""
        state = {
            "messages": [],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        
        result = router_node(state)
        
        # Should default to conversational
        assert result["query_type"] == "conversational"


@pytest.mark.unit
class TestVariantConfigs:
    """Tests for VARIANTS configuration."""
    
    def test_variant_a1_baseline(self):
        """A1 (baseline) should have all retrieval disabled."""
        
        a1 = VARIANTS.get("A1")
        assert a1 is not None
        assert a1.use_policy is False
        assert a1.use_damage is False
        assert a1.use_claims is False
        assert a1.use_router is False
        assert a1.use_retrieval is False
    
    def test_variant_a4_full(self):
        """A4 (full) should have all features enabled."""
        
        a4 = VARIANTS.get("A4")
        assert a4 is not None
        assert a4.use_policy is True
        assert a4.use_damage is True
        assert a4.use_claims is True
        assert a4.use_router is True
        assert a4.use_retrieval is True
    
    def test_all_variants_present(self):
        """All 4 variants (A1-A4) should be defined."""

        assert set(VARIANTS.keys()) == {"A1", "A2", "A3", "A4"}


# ── New tests added by code review ────────────────────────────────────────
# test_memory_node_handles_empty_messages_gracefully   (edge case)
# test_memory_node_first_type_wins_when_multiple_types (edge case 4)
# test_memory_node_does_not_overwrite_existing_incident_type (edge case)
# test_router_unknown_llm_response_defaults_conversational (coverage gap)
# ──────────────────────────────────────────────────────────────────────────

def _make_state(content="", extra=None):
    state = {
        "messages": [HumanMessage(content=content)] if content else [],
        "query_type": "conversational",
        "claim_draft": {},
        "retrieved_docs": [],
        "image_path": None,
        "detected_policy_number": None,
        "detected_insurance_type": None,
    }
    if extra:
        state.update(extra)
    return state


@pytest.mark.unit
class TestMemoryNodeEdgeCases:
    """Additional edge-case tests for memory_node."""

    def test_memory_node_handles_empty_messages_gracefully_when_no_messages(self):
        """Edge case: empty message list must not raise IndexError."""
        # Arrange
        state = _make_state("")  # produces [] messages

        # Act
        result = memory_node(state)

        # Assert — no exception; nothing extracted
        assert result["detected_insurance_type"] is None
        assert result["detected_policy_number"] is None

    def test_memory_node_first_insurance_type_wins_when_multiple_types_mentioned(self):
        """
        Edge Case 4: Message mentions motor AND home — the elif chain means
        whichever keyword appears first in the code wins (motor).
        Test documents this deterministic behaviour.
        """
        # Arrange
        state = _make_state("My car and my house were both damaged.")

        # Act
        result = memory_node(state)

        # Assert — 'motor' is the first branch matched
        assert result["detected_insurance_type"] == "motor"
        assert result["claim_draft"]["insurance_type"] == "motor"

    def test_memory_node_does_not_overwrite_existing_incident_type_when_already_set(self):
        """
        An existing incident_type in the draft must not be overwritten, because
        memory_node uses draft.get("incident_type", default).
        """
        # Arrange
        state = _make_state(
            "There was a crash at the intersection.",
            extra={"claim_draft": {"incident_type": "theft"}},
        )

        # Act
        result = memory_node(state)

        # Assert — pre-existing value wins
        assert result["claim_draft"]["incident_type"] == "theft"


@pytest.mark.unit
class TestRouterNodeEdgeCases:
    """Additional edge-case tests for router_node."""

    @patch("src.agent.llm")
    def test_router_unknown_llm_response_defaults_to_conversational_when_unrecognised(
        self, mock_llm
    ):
        """
        Coverage gap: LLM returns an unrecognised string (not one of the 4 types).
        Router should default to 'conversational', not raise KeyError.
        """
        # Arrange
        mock_llm.invoke.return_value = MagicMock(content="I cannot classify this query")
        state = _make_state("Something ambiguous that confuses the router.")

        # Act
        from src.agent import router_node
        result = router_node(state)

        # Assert
        assert result["query_type"] == "conversational"
