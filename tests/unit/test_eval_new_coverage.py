"""
tests/unit/test_eval_new_coverage.py
Extended tests for eval.py - additional coverage for edge cases and scenarios.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.eval import (
    keyword_recall,
    llm_as_judge,
    VariantConfig,
    VARIANTS,
    EvalResult,
    BENCHMARK,
    run_agent_query,
)


@pytest.mark.unit
class TestKeywordRecallAdvanced:
    """Advanced tests for keyword_recall metric."""
    
    def test_hyphenated_word_matching(self):
        """Test that hyphens are handled correctly."""
        response = "The third-party driver provided details."
        keywords = ["third party"]  # without hyphen
        
        recall = keyword_recall(response, keywords)
        
        # Should match due to hyphen normalization
        assert recall == 1.0
    
    def test_hyphenated_vs_spaced_keywords(self):
        """Test matching when keyword has hyphen but response has space."""
        response = "The pre approval form is required."
        keywords = ["pre-approval"]  # with hyphen
        
        recall = keyword_recall(response, keywords)
        
        # Should still match after normalization
        assert recall == 1.0
    
    def test_numeric_keyword_matching(self):
        """Test matching of numeric keywords."""
        response = "The excess is $750 and covers 3 categories."
        keywords = ["750", "3"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_special_characters_in_keywords(self):
        """Test handling of special characters."""
        response = "The policy covers $250-$500 excess."
        keywords = ["250", "500"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_single_keyword_list(self):
        """Test with single keyword."""
        response = "The policy covers windscreen damage."
        keywords = ["windscreen"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_very_long_keyword_list(self):
        """Test with large number of keywords."""
        response = "motor excess policy claim"
        keywords = ["motor", "excess", "policy", "claim", "damage", "cover", "denied"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == pytest.approx(4/7, abs=0.001)  # Only 4 found
    
    def test_keyword_at_boundaries(self):
        """Test keywords at start and end of response."""
        response = "excess amount is very high"
        keywords = ["excess", "high"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_punctuation_handling(self):
        """Test that punctuation doesn't prevent matching."""
        response = "Is windscreen covered? Yes, it is."
        keywords = ["windscreen", "covered"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        response = "The  excess   is   $750"
        keywords = ["excess", "750"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == 1.0
    
    def test_partial_word_no_match(self):
        """Test that substring doesn't match if not actually a substring."""
        response = "The exclusion policy is clear."
        keywords = ["exclude"]  # "exclude" != "exclusion" substring
        
        recall = keyword_recall(response, keywords)
        
        # "exclude" is NOT a substring of "exclusion" (different endings)
        assert recall == 0.0


@pytest.mark.unit
class TestLLMAsJudgeAdvanced:
    """Advanced tests for LLM-as-judge scoring."""
    
    @patch("src.eval.ollama.chat")
    def test_judge_boundary_score_1(self, mock_ollama):
        """Test judge handling of score 1 (completely wrong)."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 1, "reason": "Completely wrong"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 1
    
    @patch("src.eval.ollama.chat")
    def test_judge_boundary_score_5(self, mock_ollama):
        """Test judge handling of score 5 (perfect)."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 5, "reason": "Perfect"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 5
    
    @patch("src.eval.ollama.chat")
    def test_judge_clamp_score_above_max(self, mock_ollama):
        """Test that scores above 5 are clamped."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 10, "reason": "Too high"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 5  # Clamped to max
    
    @patch("src.eval.ollama.chat")
    def test_judge_clamp_score_below_min(self, mock_ollama):
        """Test that scores below 1 are clamped."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": -5, "reason": "Too low"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 1  # Clamped to min
    
    @patch("src.eval.ollama.chat")
    def test_judge_score_zero_clamped(self, mock_ollama):
        """Test that score 0 is clamped to 1."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 0, "reason": "Zero"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 1
    
    @patch("src.eval.ollama.chat")
    def test_judge_empty_reason_string(self, mock_ollama):
        """Test handling of empty reason."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 3, "reason": ""}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 3
        assert reason == ""
    
    @patch("src.eval.ollama.chat")
    def test_judge_very_long_reason(self, mock_ollama):
        """Test handling of very long reason."""
        long_reason = "x" * 1000
        mock_ollama.return_value = {
            "message": {"content": f'{{"score": 4, "reason": "{long_reason}"}}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 4
        assert len(reason) == 1000
    
    @patch("src.eval.ollama.chat")
    def test_judge_malformed_json_missing_score(self, mock_ollama):
        """Test handling of JSON missing score field."""
        mock_ollama.return_value = {
            "message": {"content": '{"reason": "no score"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        assert score == 3  # Default
        assert reason == "parse error"
    
    @patch("src.eval.ollama.chat")
    def test_judge_string_score_instead_of_int(self, mock_ollama):
        """Test handling of string score instead of int."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": "four", "reason": "string score"}'}
        }
        
        score, reason = llm_as_judge("q", "r", "gt")
        # Should fail to parse and return default
        assert score == 3
    
    @patch("src.eval.ollama.chat")
    def test_judge_empty_query(self, mock_ollama):
        """Test with empty query."""
        mock_ollama.return_value = {
            "message": {"content": '{"score": 3, "reason": "ok"}'}
        }
        
        score, reason = llm_as_judge("", "response", "ground_truth")
        assert score == 3
    
    @patch("src.eval.ollama.chat")
    def test_judge_very_long_query(self, mock_ollama):
        """Test with very long query."""
        long_query = "What is " + "x" * 1000
        mock_ollama.return_value = {
            "message": {"content": '{"score": 3, "reason": "ok"}'}
        }
        
        score, reason = llm_as_judge(long_query, "response", "ground_truth")
        assert mock_ollama.called
        call_messages = mock_ollama.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert long_query in prompt


@pytest.mark.unit
class TestEvalResultValidation:
    """Tests for EvalResult dataclass validation."""
    
    def test_eval_result_creation(self):
        """Test creating an EvalResult instance."""
        result = EvalResult(
            test_id="T1",
            family="factual",
            variant_key="A1",
            variant_name="baseline",
            query="What is the excess?",
            response="$750",
            latency_ms=100.0,
            input_tokens=50,
            output_tokens=15,
            total_tokens=65,
            recall_at_5=0.8,
            judge_score=4,
            judge_reason="Good answer"
        )
        
        assert result.test_id == "T1"
        assert result.family == "factual"
        assert result.recall_at_5 == 0.8
    
    def test_eval_result_zero_latency(self):
        """Test EvalResult with zero latency."""
        result = EvalResult(
            test_id="T1",
            family="factual",
            variant_key="A1",
            variant_name="baseline",
            query="Q",
            response="R",
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            recall_at_5=0.0,
            judge_score=1,
            judge_reason="Poor"
        )
        
        assert result.latency_ms == 0.0
        assert result.total_tokens == 0
    
    def test_eval_result_large_values(self):
        """Test EvalResult with large metric values."""
        result = EvalResult(
            test_id="T1",
            family="factual",
            variant_key="A1",
            variant_name="baseline",
            query="Q",
            response="R",
            latency_ms=10000.0,
            input_tokens=10000,
            output_tokens=5000,
            total_tokens=15000,
            recall_at_5=1.0,
            judge_score=5,
            judge_reason="Perfect"
        )
        
        assert result.latency_ms == 10000.0
        assert result.total_tokens == 15000
        assert result.judge_score == 5


@pytest.mark.unit
class TestBenchmarkStructure:
    """Tests for BENCHMARK data structure validation."""
    
    def test_benchmark_not_empty(self):
        """Test that BENCHMARK has test cases."""
        assert len(BENCHMARK) > 0
    
    def test_benchmark_has_all_families(self):
        """Test that BENCHMARK covers all query families."""
        families = {case["family"] for case in BENCHMARK}
        expected = {"factual", "cross_modal", "analytical", "conversational", "multi_turn"}
        
        assert families == expected
    
    def test_benchmark_unique_ids(self):
        """Test that all benchmark case IDs are unique."""
        ids = [case["id"] for case in BENCHMARK]
        assert len(ids) == len(set(ids))  # All unique
    
    def test_benchmark_required_fields(self):
        """Test that each benchmark case has required fields."""
        required_fields = {"id", "family", "query", "ground_truth", "expected_source", "keywords"}
        
        for case in BENCHMARK:
            assert all(field in case for field in required_fields)
    
    def test_benchmark_cross_modal_has_image_path(self):
        """Test that cross_modal cases have image_path or specific note."""
        cross_modal_cases = [c for c in BENCHMARK if c["family"] == "cross_modal"]
        
        for case in cross_modal_cases:
            # Either has image_path or is placeholder
            assert "image_path" in case or "photo" in case["query"].lower()
    
    def test_benchmark_multi_turn_has_prior_turns(self):
        """Test that multi_turn cases have prior_turns."""
        multi_turn_cases = [c for c in BENCHMARK if c["family"] == "multi_turn"]
        
        for case in multi_turn_cases:
            assert "prior_turns" in case
            assert isinstance(case["prior_turns"], list)
            assert len(case["prior_turns"]) > 0
    
    def test_benchmark_keywords_are_lists(self):
        """Test that keywords field is always a list."""
        for case in BENCHMARK:
            assert isinstance(case["keywords"], list)
            assert len(case["keywords"]) > 0


@pytest.mark.unit
class TestVariantFeatures:
    """Tests for variant feature interactions."""
    
    def test_a1_has_no_features(self):
        """Test A1 has no retrieval or routing."""
        a1 = VARIANTS["A1"]
        assert a1.use_retrieval is False
        assert a1.use_router is False
    
    def test_a2_has_router_but_no_damage(self):
        """Test A2 uses router and policy but not damage."""
        a2 = VARIANTS["A2"]
        assert a2.use_router is True
        assert a2.use_policy is True
        assert a2.use_damage is False
    
    def test_a3_has_all_indices_no_router(self):
        """Test A3 uses all indices but no router."""
        a3 = VARIANTS["A3"]
        assert a3.use_router is False
        assert a3.use_policy is True
        assert a3.use_damage is True
        assert a3.use_claims is True
    
    def test_a4_full_features(self):
        """Test A4 has all features enabled."""
        a4 = VARIANTS["A4"]
        assert a4.use_router is True
        assert a4.use_policy is True
        assert a4.use_damage is True
        assert a4.use_claims is True
        assert a4.use_retrieval is True
    
    def test_variant_descriptions_exist(self):
        """Test that all variants have descriptions."""
        for key, config in VARIANTS.items():
            assert config.description
            assert len(config.description) > 0


@pytest.mark.unit
class TestRunAgentQueryMocking:
    """Tests for run_agent_query function behavior."""
    
    @patch("src.eval.ollama.chat")
    def test_run_agent_query_returns_tuple(self, mock_ollama):
        """Test that run_agent_query returns proper tuple."""
        mock_ollama.return_value = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 50,
            "eval_count": 15,
        }
        
        config = VariantConfig("test", "test")
        response, in_tok, out_tok = run_agent_query("Test query", config)
        
        assert isinstance(response, str)
        assert isinstance(in_tok, int)
        assert isinstance(out_tok, int)
    
    @patch("src.eval.ollama.chat")
    def test_run_agent_query_error_returns_error_string(self, mock_ollama):
        """Test error handling returns error message."""
        mock_ollama.side_effect = Exception("Test error")
        
        config = VariantConfig("test", "test")
        response, in_tok, out_tok = run_agent_query("Test query", config)
        
        assert "ERROR" in response
        assert in_tok == 0
        assert out_tok == 0
    
    @patch("src.eval.ollama.chat")
    def test_run_agent_query_missing_token_counts(self, mock_ollama):
        """Test handling of missing token counts in response."""
        mock_ollama.return_value = {
            "message": {"content": "Test response"},
            # Missing prompt_eval_count and eval_count
        }
        
        config = VariantConfig("test", "test")
        response, in_tok, out_tok = run_agent_query("Test query", config)
        
        assert response == "Test response"
        assert in_tok == 0
        assert out_tok == 0


@pytest.mark.unit
class TestKeywordRecallDistribution:
    """Tests for keyword_recall metric edge cases."""
    
    def test_recall_exact_half(self):
        """Test exact 50% recall."""
        response = "excess policy"
        keywords = ["excess", "policy", "coverage", "claim"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == pytest.approx(0.5, rel=0.01)
    
    def test_recall_one_third(self):
        """Test one-third recall."""
        response = "excess"
        keywords = ["excess", "policy", "coverage"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == pytest.approx(1/3, rel=0.01)
    
    def test_recall_two_thirds(self):
        """Test two-thirds recall."""
        response = "excess policy"
        keywords = ["excess", "policy", "coverage"]
        
        recall = keyword_recall(response, keywords)
        
        assert recall == pytest.approx(2/3, rel=0.01)


@pytest.mark.unit
class TestVariantConfigEquality:
    """Tests for VariantConfig comparison and properties."""
    
    def test_variant_config_same_values(self):
        """Test creating identical VariantConfigs."""
        config1 = VariantConfig("test", "desc", use_policy=True, use_damage=False)
        config2 = VariantConfig("test", "desc", use_policy=True, use_damage=False)
        
        # They should have same values
        assert config1.name == config2.name
        assert config1.use_policy == config2.use_policy
    
    def test_variant_config_different_values(self):
        """Test creating different VariantConfigs."""
        config1 = VariantConfig("test1", "desc")
        config2 = VariantConfig("test2", "desc")
        
        assert config1.name != config2.name


@pytest.mark.unit
class TestAggregateResultsPrerequisites:
    """Tests for prerequisites of aggregate_results function."""
    
    def test_build_spark_df_exists(self):
        """Test that build_spark_df function exists."""
        from src.eval import build_spark_df
        assert callable(build_spark_df)
    
    def test_aggregate_results_exists(self):
        """Test that aggregate_results function exists."""
        from src.eval import aggregate_results
        assert callable(aggregate_results)
