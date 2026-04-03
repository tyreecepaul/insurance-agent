"""
tests/unit/test_eval_metrics.py
Unit tests for eval.py metrics (keyword_recall, judge parsing, etc.)
Fast, isolated tests with no external dependencies.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.eval import keyword_recall, llm_as_judge, VariantConfig, VARIANTS


@pytest.mark.unit
class TestKeywordRecall:
    """Tests for keyword_recall metric."""
    
    def test_perfect_recall(self):
        """Test perfect recall when all keywords found."""
        response = "The excess is $750 for comprehensive motor insurance. No excess applies to windscreen claims."
        keywords = ["excess", "750", "comprehensive"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_partial_recall(self):
        """Test partial recall when some keywords found."""
        response = "The excess is $750."
        keywords = ["excess", "750", "windscreen"]  # windscreen not in response
        
        recall = keyword_recall(response, keywords)
        
        assert recall == pytest.approx(2/3, rel=0.01)
    
    def test_zero_recall(self):
        """Test zero recall when no keywords found."""
        response = "Irrelevant text about other topics."
        keywords = ["excess", "windscreen", "coverage"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 0.0
    
    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        response = "The EXCESS is $750."
        keywords = ["excess", "750"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_empty_keywords(self):
        """Test with empty keyword list."""
        response = "Some response"
        keywords = []
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 0.0  # or could be 1.0 (all of 0 keywords found)
    
    def test_empty_response(self):
        """Test with empty response."""
        response = ""
        keywords = ["excess", "windscreen"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 0.0
    
    def test_keyword_substring_match(self):
        """Test that substrings count as matches."""
        response = "The excluded items are not covered."
        keywords = ["exclude"]  # "excluded" contains "exclude"
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_multiple_occurrences_counted_once(self):
        """Test that multiple occurrences still count as one match."""
        response = "excess excess excess"
        keywords = ["excess"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0


@pytest.mark.unit
class TestLLMAsJudge:
    """Tests for LLM-as-judge scoring."""
    
    @patch("src.eval.ollama.chat")
    def test_judge_perfect_score(self, mock_ollama):
        """Test judge scoring for perfect response."""
        mock_ollama.return_value = {
            "message": {
                "content": '{"score": 5, "reason": "Fully correct, evidence cited, actionable"}'
            }
        }
        
        score, reason = llm_as_judge(
            query="What is the excess?",
            response="$750 for comprehensive motor.",
            ground_truth="$750 for comprehensive motor"
        )
        
        assert score == 5
        assert "Fully correct" in reason
    
    @patch("src.eval.ollama.chat")
    def test_judge_partial_score(self, mock_ollama):
        """Test judge scoring for partially correct response."""
        mock_ollama.return_value = {
            "message": {
                "content": '{"score": 3, "reason": "Partially correct, missing key details"}'
            }
        }
        
        score, reason = llm_as_judge(
            query="What is the excess?",
            response="$750",
            ground_truth="$750 for comprehensive motor, with reduced excess options"
        )
        
        assert score == 3
        assert "missing" in reason
    
    @patch("src.eval.ollama.chat")
    def test_judge_invalid_json(self, mock_ollama):
        """Test graceful handling of invalid JSON response."""
        mock_ollama.return_value = {
            "message": {
                "content": "not valid json"
            }
        }
        
        score, reason = llm_as_judge(
            query="What is the excess?",
            response="$750",
            ground_truth="Ground truth"
        )
        
        assert score == 3  # Default fallback
        assert reason == "parse error"
    
    @patch("src.eval.ollama.chat")
    def test_judge_missing_reason_field(self, mock_ollama):
        """Test handling of JSON with missing reason field."""
        mock_ollama.return_value = {
            "message": {
                "content": '{"score": 4}'  # Missing reason
            }
        }
        
        score, reason = llm_as_judge(
            query="What is the excess?",
            response="$750 for comprehensive motor",
            ground_truth="Ground truth"
        )
        
        assert score == 4
        assert reason == ""
    
    @patch("src.eval.ollama.chat")
    def test_judge_llm_called_with_context(self, mock_ollama):
        """Test that judge is called with proper context."""
        mock_ollama.return_value = {
            "message": {
                "content": '{"score": 5, "reason": "Good"}'
            }
        }
        
        query = "What is the excess?"
        response = "$750"
        ground_truth = "Standard excess is $750"
        
        llm_as_judge(query, response, ground_truth)
        
        # Verify llm was called
        assert mock_ollama.called
        call_args = mock_ollama.call_args
        messages = call_args[1]["messages"]
        
        # Verify the prompt contains all context
        prompt_text = messages[0]["content"]
        assert query in prompt_text
        assert response in prompt_text
        assert ground_truth in prompt_text


@pytest.mark.unit
class TestVariantConfig:
    """Tests for VariantConfig dataclass."""
    
    def test_variant_config_creation(self):
        """Test that VariantConfig can be instantiated."""
        config = VariantConfig(
            name="test",
            description="Test variant",
            use_policy=True,
            use_damage=True,
            use_claims=False,
        )
        
        assert config.name == "test"
        assert config.use_policy is True
        assert config.use_damage is True
        assert config.use_claims is False
    
    def test_variant_config_defaults(self):
        """Test default values for VariantConfig."""
        config = VariantConfig(
            name="test",
            description="Test"
        )
        
        assert config.use_policy is True
        assert config.use_damage is True
        assert config.use_claims is True
        assert config.use_router is True
        assert config.use_retrieval is True


@pytest.mark.unit
class TestVariants:
    """Tests for VARIANTS configuration dictionary."""
    
    def test_variants_a1_baseline(self):
        """Test A1 (baseline) configuration."""
        a1 = VARIANTS["A1"]
        
        assert a1.name == "A1_plain_llm"
        assert a1.description == "No retrieval — baseline LLM only"
        assert a1.use_retrieval is False
    
    def test_variants_a2_text_only(self):
        """Test A2 (text-only) configuration."""
        a2 = VARIANTS["A2"]
        
        assert a2.name == "A2_text_only"
        assert a2.use_policy is True
        assert a2.use_damage is False  # No images
        assert a2.use_claims is True
    
    def test_variants_a3_no_router(self):
        """Test A3 (no router) configuration."""
        a3 = VARIANTS["A3"]
        
        assert a3.name == "A3_no_router"
        assert a3.use_router is False  # No routing
        assert a3.use_policy is True
        assert a3.use_damage is True
    
    def test_variants_a4_full(self):
        """Test A4 (full system) configuration."""
        a4 = VARIANTS["A4"]
        
        assert a4.name == "A4_full_agent"
        assert a4.use_router is True
        assert a4.use_policy is True
        assert a4.use_damage is True
        assert a4.use_claims is True
    
    def test_variants_progression(self):
        """Test that variants increase in capability A1 → A4."""
        # Count features for each variant
        feature_counts = {}
        for key in ["A1", "A2", "A3", "A4"]:
            config = VARIANTS[key]
            count = sum([
                config.use_policy,
                config.use_damage,
                config.use_claims,
                config.use_router,
                config.use_retrieval,
            ])
            feature_counts[key] = count
        
        # A1 should have fewest features
        assert feature_counts["A1"] == 0
        
        # A4 should have most features
        assert feature_counts["A4"] == 5
        
        # Progressive increase (not strictly monotonic due to design)
        # but A4 > A1 definitely
        assert feature_counts["A4"] > feature_counts["A1"]
