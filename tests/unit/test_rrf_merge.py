"""
tests/unit/test_rrf_merge.py
Unit tests for tools._rrf_merge hybrid-retrieval fusion.
"""

import pytest
from src.tools import _rrf_merge, RetrievalResult


@pytest.mark.unit
class TestRRFMerge:
    """Tests for Reciprocal Rank Fusion merge logic."""

    # ── MEDIUM: Fix _rrf_merge to include BM25-only results ──────────────────
    def test_bm25_only_doc_included_when_not_in_dense_results(self):
        """
        MEDIUM - Fix _rrf_merge so BM25-only docs are not silently dropped.
        A document that scores well in BM25 but missed the dense top-k should
        still appear in the merged output.
        """
        # Arrange
        dense_ids   = ["doc-A"]
        bm25_ids    = ["doc-B", "doc-A"]   # doc-B is BM25-only
        dense_docs  = ["Dense content A"]
        dense_metas = [{"insurance_type": "motor"}]
        dense_dists = [0.1]
        bm25_docs   = ["BM25 content B", "Dense content A"]
        bm25_metas  = [{"insurance_type": "home"}, {"insurance_type": "motor"}]

        # Act
        results = _rrf_merge(
            dense_ids, bm25_ids,
            dense_docs, dense_metas, dense_dists,
            source="policy",
            bm25_docs=bm25_docs,
            bm25_metas=bm25_metas,
        )
        result_ids = [r.doc_id for r in results]

        # Assert
        assert "doc-B" in result_ids, (
            "BM25-only doc 'doc-B' was dropped. "
            "Hybrid retrieval degrades to dense-only."
        )
        assert "doc-A" in result_ids, "Dense doc should also be present"

    def test_bm25_only_doc_has_correct_content_when_included(self):
        """BM25-only result should carry its own content, not a blank string."""
        dense_ids   = ["doc-A"]
        bm25_ids    = ["doc-B", "doc-A"]
        dense_docs  = ["Content A"]
        dense_metas = [{}]
        dense_dists = [0.2]
        bm25_docs   = ["Content B", "Content A"]
        bm25_metas  = [{"page": 3}, {}]

        results = _rrf_merge(
            dense_ids, bm25_ids,
            dense_docs, dense_metas, dense_dists,
            source="policy",
            bm25_docs=bm25_docs,
            bm25_metas=bm25_metas,
        )
        b_result = next((r for r in results if r.doc_id == "doc-B"), None)
        assert b_result is not None
        assert b_result.content == "Content B"
        assert b_result.metadata == {"page": 3}

    def test_dense_only_doc_still_present_when_not_in_bm25(self):
        """Dense-only docs should still appear (existing behaviour must not regress)."""
        dense_ids   = ["doc-X", "doc-Y"]
        bm25_ids    = ["doc-Y"]          # doc-X is dense-only
        dense_docs  = ["Content X", "Content Y"]
        dense_metas = [{}, {}]
        dense_dists = [0.05, 0.15]

        results = _rrf_merge(
            dense_ids, bm25_ids,
            dense_docs, dense_metas, dense_dists,
            source="policy",
        )
        result_ids = [r.doc_id for r in results]
        assert "doc-X" in result_ids
        assert "doc-Y" in result_ids

    def test_rrf_scores_are_positive_for_all_results(self):
        """Every returned result should have a positive RRF score."""
        dense_ids   = ["a", "b"]
        bm25_ids    = ["b", "c"]
        dense_docs  = ["Doc A", "Doc B"]
        dense_metas = [{}, {}]
        dense_dists = [0.1, 0.2]
        bm25_docs   = ["Doc B", "Doc C"]
        bm25_metas  = [{}, {}]

        results = _rrf_merge(
            dense_ids, bm25_ids,
            dense_docs, dense_metas, dense_dists,
            source="claims",
            bm25_docs=bm25_docs,
            bm25_metas=bm25_metas,
        )
        assert all(r.score > 0 for r in results)

    def test_both_lists_empty_returns_empty(self):
        """Empty dense and BM25 inputs should return empty results."""
        results = _rrf_merge([], [], [], [], [], source="policy")
        assert results == []

    def test_source_label_is_propagated_to_all_results(self):
        """All results should carry the source label passed to _rrf_merge."""
        dense_ids   = ["d1"]
        bm25_ids    = ["d1"]
        results = _rrf_merge(
            dense_ids, bm25_ids,
            ["Content"], [{}], [0.1],
            source="damage",
        )
        assert all(r.source == "damage" for r in results)
