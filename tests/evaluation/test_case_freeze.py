from src.rag_platform.evaluation.case_persistence import (
    build_frozen_jsonl,
    frozen_content_sha256,
    validate_required_evidence_mapped,
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


def _case(code: str, mapping_status: MappingStatus) -> ReviewedEvalCase:
    evidence_kwargs = {
        "source_doc_code": "FAQ_ORDER_STATUS_001",
        "evidence_quote": "待支付订单可由用户直接取消。",
        "fact_key": "order_cancel_pending_payment",
        "relevance_grade": 3,
        "mapping_status": mapping_status,
    }
    if mapping_status == MappingStatus.MAPPED:
        evidence_kwargs.update(mapped_doc_id=11, mapped_chunk_id=101)
    return ReviewedEvalCase(
        case_code=code,
        question=f"{code}：待支付订单如何取消？",
        reference_answer="待支付订单可直接取消。",
        case_type=EvalCaseType.DIRECT,
        expected_action=ExpectedAction.ANSWER,
        dataset_split=DatasetSplit.DEVELOPMENT,
        required_fact_count=1,
        generation_metadata={"source_group": "topic:order"},
        evidences=[EvidenceSpec(**evidence_kwargs)],
        review_status=ReviewStatus.PASSED,
        review_score=0.95,
    )


def test_frozen_jsonl_is_sorted_and_hash_is_reproducible() -> None:
    first = _case("CASE_DIRECT_001", MappingStatus.MAPPED)
    second = _case("CASE_DIRECT_002", MappingStatus.MAPPED)

    content_a = build_frozen_jsonl([second, first])
    content_b = build_frozen_jsonl([first, second])

    assert content_a == content_b
    assert content_a.splitlines()[0].find("CASE_DIRECT_001") >= 0
    assert frozen_content_sha256(content_a) == frozen_content_sha256(content_b)


def test_freeze_rejects_unmapped_required_evidence() -> None:
    errors = validate_required_evidence_mapped(
        [_case("CASE_DIRECT_001", MappingStatus.MISSING)]
    )

    assert errors == ["CASE_DIRECT_001存在未映射的必要证据"]
