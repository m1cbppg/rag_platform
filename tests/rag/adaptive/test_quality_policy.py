from types import SimpleNamespace

from src.rag_platform.rag.adaptive.models import (
    RetrievalQualityFeatures,
    RetryStrategy,
)
from src.rag_platform.rag.adaptive.quality_policy import (
    RetrievalQualityPolicy,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        adaptive_quality_good_threshold=0.68,
        adaptive_quality_poor_threshold=0.25,
        adaptive_rerank_top1_threshold=0.55,
        adaptive_rerank_top3_threshold=0.50,
        adaptive_min_candidate_count=3,
        adaptive_min_distinct_documents=2,
        adaptive_min_version_count=2,
    )


def _features(**overrides) -> RetrievalQualityFeatures:
    values = {
        "candidate_count": 10,
        "distinct_document_count": 4,
        "channel_overlap_at_10": 0.4,
        "rerank_top1": 0.85,
        "rerank_top3_mean": 0.75,
        "rerank_margin": 0.1,
        "target_type_coverage": 1.0,
        "exact_terms": [],
        "exact_term_coverage": 1.0,
        "distinct_version_count": 1,
        "comparison_intent": False,
    }
    values.update(overrides)
    return RetrievalQualityFeatures(**values)


def test_no_candidates_is_poor_and_relaxes_filter() -> None:
    decision = RetrievalQualityPolicy(_settings()).decide(
        _features(
            candidate_count=0,
            distinct_document_count=0,
            rerank_top1=0.0,
            rerank_top3_mean=0.0,
        )
    )

    assert decision.level == "POOR"
    assert decision.retry_strategy == RetryStrategy.RELAX_FILTER


def test_missing_exact_term_forces_bm25_even_when_rerank_is_high() -> None:
    decision = RetrievalQualityPolicy(_settings()).decide(
        _features(
            exact_terms=["F-ORDER-001"],
            exact_term_coverage=0.0,
            rerank_top1=0.95,
        )
    )

    assert decision.level == "WEAK"
    assert decision.retry_strategy == RetryStrategy.FORCE_BM25
    assert "精确词" in " ".join(decision.reasons)


def test_comparison_with_one_version_triggers_query_rewrite() -> None:
    decision = RetrievalQualityPolicy(_settings()).decide(
        _features(
            comparison_intent=True,
            distinct_version_count=1,
        )
    )

    assert decision.level == "WEAK"
    assert decision.retry_strategy == RetryStrategy.QUERY_REWRITE
    assert "版本" in " ".join(decision.reasons)


def test_low_rerank_confidence_triggers_query_rewrite() -> None:
    decision = RetrievalQualityPolicy(_settings()).decide(
        _features(
            rerank_top1=0.50,
            rerank_top3_mean=0.45,
        )
    )

    assert decision.level == "WEAK"
    assert decision.retry_strategy == RetryStrategy.QUERY_REWRITE


def test_strong_evidence_is_good_without_retry() -> None:
    decision = RetrievalQualityPolicy(_settings()).decide(_features())

    assert decision.level == "GOOD"
    assert decision.retry_strategy == RetryStrategy.NONE
