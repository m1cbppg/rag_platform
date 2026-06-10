import pytest

from src.rag_platform.evaluation.retrieval_metrics import (
    fact_coverage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_unique_gold_chunks() -> None:
    assert recall_at_k([101, 101, 102, 999], [101, 102], 1) == 0.5
    assert recall_at_k([101, 101, 102, 999], [101, 102], 3) == 1.0


def test_recall_at_k_returns_zero_when_gold_is_not_retrieved() -> None:
    assert recall_at_k([201, 202], [101, 102], 5) == 0.0


def test_recall_at_k_is_not_applicable_without_gold_chunks() -> None:
    assert recall_at_k([201, 202], [], 5) is None


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k必须大于0"):
        recall_at_k([101], [101], 0)


def test_reciprocal_rank_uses_first_unique_gold_position() -> None:
    assert reciprocal_rank([999, 999, 101, 102], [101, 102]) == 0.5


def test_reciprocal_rank_returns_zero_for_a_miss_and_none_without_gold() -> None:
    assert reciprocal_rank([201, 202], [101]) == 0.0
    assert reciprocal_rank([201, 202], []) is None


def test_ndcg_uses_graded_relevance_and_ignores_duplicate_results() -> None:
    relevance = {101: 3, 102: 2, 103: 1}

    ideal = ndcg_at_k([101, 102, 103], relevance, 3)
    reordered = ndcg_at_k([103, 101, 101, 102], relevance, 3)

    assert ideal == pytest.approx(1.0)
    assert reordered == pytest.approx(0.736364, abs=1e-6)


def test_ndcg_returns_zero_for_a_miss_and_none_without_gold() -> None:
    assert ndcg_at_k([201, 202], {101: 3}, 5) == 0.0
    assert ndcg_at_k([201, 202], {}, 5) is None


def test_fact_coverage_counts_distinct_required_facts() -> None:
    fact_keys_by_chunk = {
        101: {"refund_window"},
        102: {"coupon_return"},
        103: {"risk_exception", "refund_window"},
    }

    assert fact_coverage([101, 101, 103], fact_keys_by_chunk) == pytest.approx(
        2 / 3
    )


def test_fact_coverage_returns_zero_for_a_miss_and_none_without_facts() -> None:
    assert fact_coverage([201], {101: {"refund_window"}}) == 0.0
    assert fact_coverage([201], {}) is None
