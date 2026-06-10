from src.rag_platform.evaluation.evidence_mapper import (
    EvidenceMapper,
    map_case_evidence,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
)


def _case(quote: str) -> ReviewedEvalCase:
    return ReviewedEvalCase(
        case_code="CASE_DIRECT_001",
        question="待支付订单如何取消？",
        reference_answer="待支付订单可直接取消。",
        case_type=EvalCaseType.DIRECT,
        expected_action=ExpectedAction.ANSWER,
        dataset_split=DatasetSplit.DEVELOPMENT,
        required_fact_count=1,
        generation_metadata={"source_group": "topic:order"},
        evidences=[
            EvidenceSpec(
                source_doc_code="FAQ_ORDER_STATUS_001",
                evidence_quote=quote,
                fact_key="order_cancel_pending_payment",
                relevance_grade=3,
            )
        ],
        review_status=ReviewStatus.PASSED,
        review_score=0.95,
    )


def test_mapper_prefers_unique_exact_match() -> None:
    result = EvidenceMapper().map_quote(
        quote="待支付订单可由用户直接取消。",
        chunks=[
            {"id": 101, "content": "规则：待支付订单可由用户直接取消。"},
            {"id": 102, "content": "已发货订单应走售后流程。"},
        ],
    )

    assert result.status == MappingStatus.MAPPED
    assert result.chunk_id == 101
    assert result.method == "EXACT"


def test_mapper_supports_punctuation_normalization() -> None:
    result = EvidenceMapper().map_quote(
        quote="退款失败时，需要人工核查。",
        chunks=[
            {"id": 201, "content": "退款失败时需要人工核查"},
        ],
    )

    assert result.status == MappingStatus.MAPPED
    assert result.chunk_id == 201
    assert result.method == "PUNCTUATION_NORMALIZED"


def test_mapper_marks_duplicate_matches_as_ambiguous() -> None:
    result = EvidenceMapper().map_quote(
        quote="订单可以取消",
        chunks=[
            {"id": 301, "content": "订单可以取消"},
            {"id": 302, "content": "订单可以取消"},
        ],
    )

    assert result.status == MappingStatus.AMBIGUOUS
    assert result.chunk_id is None


def test_mapper_prefers_child_chunk_over_matching_overview() -> None:
    result = EvidenceMapper().map_quote(
        quote="确认多余流水未绑定其他订单",
        chunks=[
            {
                "id": 177,
                "parent_chunk_id": None,
                "content": "总览：确认多余流水未绑定其他订单",
            },
            {
                "id": 179,
                "parent_chunk_id": 177,
                "content": "步骤：确认多余流水未绑定其他订单",
            },
        ],
    )

    assert result.status == MappingStatus.MAPPED
    assert result.chunk_id == 179
    assert result.method == "EXACT_MOST_SPECIFIC"


def test_mapper_reports_cross_chunk_quote_without_silent_selection() -> None:
    result = EvidenceMapper().map_quote(
        quote="商家确认后进入退款流程",
        chunks=[
            {"id": 401, "content": "商家确认后"},
            {"id": 402, "content": "进入退款流程"},
        ],
    )

    assert result.status == MappingStatus.AMBIGUOUS
    assert "401" in result.reason
    assert "402" in result.reason


def test_map_case_evidence_sets_document_and_chunk_ids() -> None:
    mapped = map_case_evidence(
        case=_case("待支付订单可由用户直接取消。"),
        source_documents={
            "FAQ_ORDER_STATUS_001": {
                "mapped_doc_id": 11,
            }
        },
        chunks_by_source_code={
            "FAQ_ORDER_STATUS_001": [
                {"id": 101, "content": "待支付订单可由用户直接取消。"}
            ]
        },
    )

    evidence = mapped.evidences[0]
    assert evidence.mapping_status == MappingStatus.MAPPED
    assert evidence.mapped_doc_id == 11
    assert evidence.mapped_chunk_id == 101
