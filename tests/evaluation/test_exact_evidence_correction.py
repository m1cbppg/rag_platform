import pytest

from src.rag_platform.evaluation.exact_evidence_correction import (
    correct_exact_identifier_evidence,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    Difficulty,
    EvalCaseType,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
    SourceDocumentType,
)


def _case(
    *,
    code: str,
    case_type: EvalCaseType = EvalCaseType.EXACT,
    identifier: str | None = "F-ORDER-001",
    evidence_chunk_id: int = 54,
    evidence_quote: str = "已发货订单不能直接取消，应走售后流程",
) -> ReviewedEvalCase:
    metadata = {"source_doc_codes": ["FAQ_ORDER_STATUS_001"]}
    if identifier is not None:
        metadata["required_identifier"] = identifier
    return ReviewedEvalCase(
        case_code=code,
        question="收到错误码F-ORDER-001时应该怎么处理？",
        normalized_question="收到错误码forder001时应该怎么处理",
        reference_answer="已发货订单不能直接取消，应走售后流程。",
        case_type=case_type,
        target_doc_types=[SourceDocumentType.FAQ],
        expected_action=ExpectedAction.ANSWER,
        difficulty=Difficulty.EASY,
        dataset_split=DatasetSplit.DEVELOPMENT,
        business_domain="ecommerce_after_sales",
        required_fact_count=1,
        generation_metadata=metadata,
        evidences=[
            EvidenceSpec(
                source_doc_code="FAQ_ORDER_STATUS_001",
                evidence_quote=evidence_quote,
                fact_key="order_cancel_shipped",
                relevance_grade=3,
                mapped_doc_id=15,
                mapped_chunk_id=evidence_chunk_id,
                mapping_status=MappingStatus.MAPPED,
                mapping_reason="原始映射",
            )
        ],
        review_status=ReviewStatus.PASSED,
        review_score=0.99,
        review_reason="通过",
    )


SOURCE_DOCUMENTS = [
    {
        "source_doc_code": "FAQ_ORDER_STATUS_001",
        "mapped_doc_id": 15,
    }
]

CHUNKS_BY_DOC_ID = {
    15: [
        {
            "id": 54,
            "doc_id": 15,
            "sort_order": 5,
            "title": "已发货订单如何取消？",
            "content": (
                "问题：已发货订单如何取消？\n"
                "答案：已发货订单不能直接取消，应走售后流程。"
            ),
        },
        {
            "id": 57,
            "doc_id": 15,
            "sort_order": 8,
            "title": "提示F-ORDER-001是什么意思？",
            "content": (
                "问题：提示F-ORDER-001是什么意思？\n"
                "答案：错误码F-ORDER-001表示当前订单状态不允许取消。"
            ),
        },
    ]
}


def test_adds_identifier_chunk_as_independent_required_fact() -> None:
    corrected, report = correct_exact_identifier_evidence(
        cases=[_case(code="CASE_EXACT_001")],
        source_documents=SOURCE_DOCUMENTS,
        chunks_by_doc_id=CHUNKS_BY_DOC_ID,
    )

    case = corrected[0]
    assert case.required_fact_count == 2
    assert [item.mapped_chunk_id for item in case.evidences] == [54, 57]
    added = case.evidences[1]
    assert added.fact_key == "required_identifier_f_order_001"
    assert added.relevance_grade == 3
    assert "F-ORDER-001" in added.evidence_quote
    assert report["corrected_case_count"] == 1
    assert report["corrections"][0]["added_chunk_id"] == 57
    assert case.generation_metadata["evidence_correction"][
        "version"
    ] == "exact-identifier-v1"


def test_keeps_case_unchanged_when_identifier_is_already_in_gold() -> None:
    corrected, report = correct_exact_identifier_evidence(
        cases=[
            _case(
                code="CASE_EXACT_002",
                evidence_chunk_id=57,
                evidence_quote=(
                    "错误码F-ORDER-001表示当前订单状态不允许取消"
                ),
            )
        ],
        source_documents=SOURCE_DOCUMENTS,
        chunks_by_doc_id=CHUNKS_BY_DOC_ID,
    )

    assert corrected[0].required_fact_count == 1
    assert len(corrected[0].evidences) == 1
    assert report["corrected_case_count"] == 0


def test_does_not_modify_non_exact_case() -> None:
    corrected, report = correct_exact_identifier_evidence(
        cases=[
            _case(
                code="CASE_DIRECT_001",
                case_type=EvalCaseType.DIRECT,
            )
        ],
        source_documents=SOURCE_DOCUMENTS,
        chunks_by_doc_id=CHUNKS_BY_DOC_ID,
    )

    assert corrected[0].required_fact_count == 1
    assert report["corrected_case_count"] == 0


def test_fails_when_identifier_cannot_be_found_in_declared_sources() -> None:
    with pytest.raises(ValueError, match="CASE_EXACT_003"):
        correct_exact_identifier_evidence(
            cases=[
                _case(
                    code="CASE_EXACT_003",
                    identifier="UNKNOWN-CODE-999",
                )
            ],
            source_documents=SOURCE_DOCUMENTS,
            chunks_by_doc_id=CHUNKS_BY_DOC_ID,
        )
