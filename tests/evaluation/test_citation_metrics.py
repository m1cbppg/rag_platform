from src.rag_platform.evaluation.citation_metrics import (
    citation_precision,
    citation_recall,
    evaluate_citations,
)


def test_citation_metrics_deduplicate_gold_and_cited_chunks() -> None:
    result = evaluate_citations(
        cited_chunk_ids=[101, 101, 999],
        gold_chunk_ids=[101, 102, 102],
    )

    assert result.true_positive_count == 1
    assert result.cited_count == 2
    assert result.gold_count == 2
    assert result.precision == 0.5
    assert result.recall == 0.5


def test_citation_metrics_report_correct_but_incomplete_citations() -> None:
    result = evaluate_citations(
        cited_chunk_ids=[101],
        gold_chunk_ids=[101, 102],
    )

    assert result.precision == 1.0
    assert result.recall == 0.5


def test_missing_citations_have_zero_recall_and_undefined_precision() -> None:
    result = evaluate_citations(
        cited_chunk_ids=[],
        gold_chunk_ids=[101],
    )

    assert result.precision is None
    assert result.recall == 0.0


def test_wrong_citations_have_zero_precision_and_recall() -> None:
    result = evaluate_citations(
        cited_chunk_ids=[999],
        gold_chunk_ids=[101],
    )

    assert result.precision == 0.0
    assert result.recall == 0.0


def test_no_answer_case_without_gold_is_not_applicable() -> None:
    result = evaluate_citations(
        cited_chunk_ids=[],
        gold_chunk_ids=[],
    )

    assert result.precision is None
    assert result.recall is None
    assert result.gold_count == 0


def test_precision_and_recall_helpers_match_combined_result() -> None:
    assert citation_precision([101, 999], [101, 102]) == 0.5
    assert citation_recall([101, 999], [101, 102]) == 0.5
