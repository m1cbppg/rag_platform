from src.rag_platform.evaluation.metric_calculator import (
    build_gold_annotations,
    calculate_case_metrics,
)
from src.rag_platform.evaluation.models import (
    ActualAction,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
)


def test_calculator_builds_complete_answer_case_metrics() -> None:
    result = calculate_case_metrics(
        retrieved_chunk_ids=[999, 101, 102],
        relevance_by_chunk={101: 3, 102: 3},
        fact_keys_by_chunk={
            101: {"refund_window"},
            102: {"coupon_return"},
        },
        cited_chunk_ids=[101],
        expected_action=ExpectedAction.ANSWER,
        actual_action=ActualAction.ANSWER,
        retrieval_rounds=2,
    )

    assert result.recall_at_1 == 0.0
    assert result.recall_at_3 == 1.0
    assert result.reciprocal_rank == 0.5
    assert result.ndcg_at_5 is not None
    assert result.fact_coverage == 1.0
    assert result.citation_precision == 1.0
    assert result.citation_recall == 0.5
    assert result.action_correct is True
    assert result.retrieval_rounds == 2


def test_calculator_marks_retrieval_and_citation_metrics_not_applicable_for_no_answer() -> None:
    result = calculate_case_metrics(
        retrieved_chunk_ids=[999],
        relevance_by_chunk={},
        fact_keys_by_chunk={},
        cited_chunk_ids=[],
        expected_action=ExpectedAction.REFUSE,
        actual_action=ActualAction.REFUSE,
    )

    assert result.recall_at_1 is None
    assert result.reciprocal_rank is None
    assert result.ndcg_at_5 is None
    assert result.fact_coverage is None
    assert result.citation_precision is None
    assert result.citation_recall is None
    assert result.action_correct is True


def test_gold_builder_merges_fact_keys_for_the_same_chunk() -> None:
    annotations = build_gold_annotations(
        [
            EvidenceSpec(
                source_doc_code="RULE_REFUND_001",
                evidence_quote="退款时限规则。",
                fact_key="refund_window",
                relevance_grade=2,
                mapped_doc_id=10,
                mapped_chunk_id=101,
                mapping_status=MappingStatus.MAPPED,
            ),
            EvidenceSpec(
                source_doc_code="RULE_REFUND_001",
                evidence_quote="退款例外规则。",
                fact_key="refund_exception",
                relevance_grade=3,
                mapped_doc_id=10,
                mapped_chunk_id=101,
                mapping_status=MappingStatus.MAPPED,
            ),
        ]
    )

    assert annotations.relevance_by_chunk == {101: 3}
    assert annotations.fact_keys_by_chunk == {
        101: {"refund_window", "refund_exception"}
    }
