"""
tests/integration/test_eval_e2e.py
End-to-end tests for eval.py with mocked Ollama and PySpark.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.eval import (
    run_agent_query,
    llm_as_judge,
    build_spark_df,
    aggregate_results,
    VARIANTS,
    EvalResult,
)


@pytest.mark.integration
class TestRunAgentQuery:
    """Integration tests for run_agent_query function."""
    
    @patch("src.eval.ollama.chat")
    def test_run_agent_query_baseline(self, mock_ollama):
        """Test baseline (A1) query — no retrieval."""
        mock_ollama.return_value = {
            "message": {"content": "The excess is $750."},
            "prompt_eval_count": 50,
            "eval_count": 15,
        }
        
        config = VARIANTS["A1"]
        response, in_tok, out_tok = run_agent_query(
            query="What is the excess?",
            config=config,
        )
        
        assert response == "The excess is $750."
        assert in_tok == 50
        assert out_tok == 15
    
    @patch("src.eval.ollama.chat")
    @patch("src.eval.search_policy")
    def test_run_agent_query_with_retrieval(self, mock_search, mock_ollama):
        """Test A2 (text-only) — with policy retrieval."""
        from tests.conftest import MockRetrievalResult
        
        # Mock policy search
        mock_search.return_value = [
            MockRetrievalResult(
                source="policy",
                doc_id="POL-001",
                content="Excess: $750 for comprehensive motor",
                metadata={"page_number": 5}
            )
        ]
        
        mock_ollama.return_value = {
            "message": {"content": "Based on policy, excess is $750."},
            "prompt_eval_count": 200,
            "eval_count": 50,
        }
        
        config = VARIANTS["A2"]
        response, in_tok, out_tok = run_agent_query(
            query="What is the excess for motor?",
            config=config,
        )
        
        assert response == "Based on policy, excess is $750."
        assert mock_search.called
        # A2 should use more tokens than A1 due to context
        assert in_tok > 100
    
    @patch("src.eval.ollama.chat")
    def test_run_agent_query_error_handling(self, mock_ollama):
        """Test that errors are handled gracefully."""
        mock_ollama.side_effect = Exception("API error")
        
        config = VARIANTS["A1"]
        response, in_tok, out_tok = run_agent_query(
            query="What is the excess?",
            config=config,
        )
        
        # Should return error message
        assert "ERROR" in response or response == "ERROR: API error"


@pytest.mark.integration
class TestEvalDataFrame:
    """Integration tests for PySpark DataFrame operations."""
    
    def test_build_spark_df(self, spark_session):
        """Test building Spark DataFrame from EvalResults."""
        results = [
            EvalResult(
                test_id="T1", family="factual", variant_key="A1", variant_name="baseline",
                query="What is the excess?", response="$750",
                latency_ms=100.0, input_tokens=50, output_tokens=15, total_tokens=65,
                recall_at_5=1.0, judge_score=5, judge_reason="Correct"
            ),
            EvalResult(
                test_id="T2", family="factual", variant_key="A1", variant_name="baseline",
                query="Is windscreen covered?", response="Yes, no excess",
                latency_ms=120.0, input_tokens=60, output_tokens=12, total_tokens=72,
                recall_at_5=0.8, judge_score=4, judge_reason="Minor gap"
            ),
        ]
        
        df = build_spark_df(results, spark_session)
        
        assert len(df) == 2  # pandas DataFrame uses len(), not count()
        assert len(df.columns) == 13
    
    def test_aggregate_results(self, spark_session):
        """Test aggregation of results."""
        results = [
            # A1 results
            EvalResult("T1", "factual", "A1", "baseline", "q1", "r1", 100.0, 50, 15, 65, 1.0, 5, "good"),
            EvalResult("T2", "factual", "A1", "baseline", "q2", "r2", 110.0, 55, 16, 71, 0.8, 4, "ok"),
            # A4 results
            EvalResult("T3", "factual", "A4", "full", "q3", "r3", 200.0, 150, 50, 200, 1.0, 5, "good"),
            EvalResult("T4", "cross_modal", "A4", "full", "q4", "r4", 250.0, 180, 60, 240, 0.9, 4, "ok"),
        ]
        
        df = build_spark_df(results, spark_session)
        variant_summary, family_breakdown, cross_modal_gap = aggregate_results(df)
        
        # Check variant summary (now pandas DataFrame)
        summary_rows = variant_summary if isinstance(variant_summary, list) else variant_summary.to_dict('records')
        assert len(summary_rows) == 2  # A1 and A4
        
        a1_row = [r for r in summary_rows if r["variant_key"] == "A1"][0]
        assert a1_row["n_tests"] == 2
        assert float(a1_row["avg_recall"]) == pytest.approx(0.9, rel=0.05)
        
        # Check that A4 has higher latency (has retrieval)
        a4_row = [r for r in summary_rows if r["variant_key"] == "A4"][0]
        assert float(a4_row["avg_latency_ms"]) > float(a1_row["avg_latency_ms"])
    
    def test_family_breakdown(self, spark_session):
        """Test per-family metric breakdown."""
        results = [
            EvalResult("T1", "factual", "A1", "baseline", "q1", "r1", 100.0, 50, 15, 65, 1.0, 5, "good"),
            EvalResult("T2", "cross_modal", "A1", "baseline", "q2", "r2", 110.0, 55, 16, 71, 0.8, 4, "ok"),
        ]
        
        df = build_spark_df(results, spark_session)
        _, family_breakdown, _ = aggregate_results(df)
        
        fb_rows = family_breakdown if isinstance(family_breakdown, list) else family_breakdown.to_dict('records')
        families = {row["family"] for row in fb_rows}
        
        assert "factual" in families
        assert "cross_modal" in families


@pytest.mark.integration
class TestEvalIntegration:
    """Integration tests combining multiple eval components."""
    
    @patch("src.eval.ollama.chat")
    def test_eval_single_test_case(self, mock_ollama):
        """Test full eval pipeline for single test case."""
        mock_ollama.return_value = {
            "message": {"content": "The excess is $750."},
            "prompt_eval_count": 100,
            "eval_count": 25,
        }
        
        config = VARIANTS["A1"]
        response, in_tok, out_tok = run_agent_query(
            query="What is the excess?",
            config=config,
        )
        
        # Should have tokens populated
        assert in_tok > 0
        assert out_tok > 0
        assert len(response) > 0
    
    @patch("src.eval.ollama.chat")
    def test_eval_result_dataclass(self, mock_ollama):
        """Test creating and serializing EvalResult."""
        result = EvalResult(
            test_id="T1",
            family="factual",
            variant_key="A1",
            variant_name="baseline",
            query="What is the excess?",
            response="$750",
            latency_ms=100.5,
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
            recall_at_5=1.0,
            judge_score=5,
            judge_reason="Correct"
        )
        
        # Verify all fields are set
        assert result.test_id == "T1"
        assert result.total_tokens == 125
        assert result.recall_at_5 == 1.0
        
        # Should be JSON-serializable
        import json
        from dataclasses import asdict
        json_str = json.dumps(asdict(result), default=str)
        assert "factual" in json_str
